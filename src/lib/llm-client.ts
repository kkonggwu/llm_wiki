import type { LlmConfig } from "@/stores/wiki-store"
import { isAzureOpenAiEndpoint } from "@/lib/azure-openai"
import { getProviderConfig, type RequestOverrides } from "./llm-providers"
import { getHttpFetch, isFetchNetworkError } from "./tauri-fetch"
import { countReasoningCharsInLine, extractReasoningTextFromLine } from "./reasoning-detector"

export type { ChatMessage, ContentBlock, RequestOverrides } from "./llm-providers"
export { isFetchNetworkError } from "./tauri-fetch"

export interface StreamCallbacks {
  onToken: (token: string) => void
  onReasoningToken?: (token: string) => void
  onDone: () => void
  onError: (error: Error) => void
}

function bufferedStreamCallbacks(callbacks: StreamCallbacks): StreamCallbacks {
  let content = ""
  let reasoning = ""
  return {
    onToken: (token) => { content += token },
    onReasoningToken: (token) => { reasoning += token },
    onDone: () => {
      if (reasoning) callbacks.onReasoningToken?.(reasoning)
      if (content) callbacks.onToken(content)
      callbacks.onDone()
    },
    onError: callbacks.onError,
  }
}

// Lazy import keeps the Tauri event/invoke bindings out of bundles that
// never touch the subprocess provider (e.g. vitest with a fetch mock).
async function streamViaClaudeCodeCli(
  config: LlmConfig,
  messages: import("./llm-providers").ChatMessage[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
  requestOverrides?: RequestOverrides,
) {
  const mod = await import("./claude-cli-transport")
  return mod.streamClaudeCodeCli(config, messages, callbacks, signal, requestOverrides)
}

async function streamViaCodexCli(
  config: LlmConfig,
  messages: import("./llm-providers").ChatMessage[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
  requestOverrides?: RequestOverrides,
) {
  const mod = await import("./codex-cli-transport")
  return mod.streamCodexCli(config, messages, callbacks, signal, requestOverrides)
}

function parseLines(
  decoder: TextDecoder,
  chunk: Uint8Array,
  buffer: string,
): [string[], string] {
  const text = buffer + decoder.decode(chunk, { stream: true })
  const lines = text.split("\n")
  const remaining = lines.pop() ?? ""
  return [lines, remaining]
}

interface EndpointErrorEnvelope {
  error?: {
    code?: string | number
    message?: string
  } | string
}

function parseEndpointErrorEnvelope(record: string): Error | null {
  const payload = record.startsWith("data:")
    ? record.slice(5).trim()
    : record

  if (!payload.startsWith("{")) return null

  try {
    const parsed = JSON.parse(payload) as EndpointErrorEnvelope
    const message = typeof parsed.error === "string"
      ? parsed.error
      : parsed.error?.message
    if (!message) return null

    const code = typeof parsed.error === "object" && parsed.error?.code !== undefined
      ? ` ${parsed.error.code}`
      : ""
    return new Error(`LLM endpoint error${code}: ${message}`)
  } catch {
    return null
  }
}

function splitFinalStreamRecords(text: string): string[] {
  // Some local transports expose a fully buffered SSE body with escaped
  // record separators. Only split after a complete JSON SSE record; a model
  // response can legitimately contain the text "\n\ndata:", and splitting
  // that sequence while it is still inside a JSON string would corrupt it.
  if (/[\r\n]/.test(text) || !/^\s*data:/.test(text)) {
    return text.split(/\r?\n/)
  }

  const records: string[] = []
  const separator = /(?:\\r)?\\n(?:\\r)?\\n(?=data:)/g
  let recordStart = 0
  let match: RegExpExecArray | null

  while ((match = separator.exec(text)) !== null) {
    const candidate = text.slice(recordStart, match.index).trim()
    const payload = candidate.startsWith("data:")
      ? candidate.slice(5).trim()
      : ""
    let complete = payload === "[DONE]"
    if (!complete && payload.startsWith("{")) {
      try {
        JSON.parse(payload)
        complete = true
      } catch {
        // The separator-like text is inside an incomplete JSON string.
      }
    }
    if (complete) {
      records.push(candidate)
      recordStart = match.index + match[0].length
    }
  }

  records.push(text.slice(recordStart))
  return records
}

function isRequestCancelledError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err)
  return /^request cancel(?:l)?ed$/i.test(message.trim())
}

export function isReasoningOnlyResponseError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err)
  return /^Model produced [\d,]+ characters of reasoning \/ chain-of-thought, but no actual response content\./.test(message)
}

function shouldRetryWithoutTemperature(
  config: LlmConfig,
  status: number,
  errorDetail: string,
  requestOverrides?: RequestOverrides,
): boolean {
  if (config.provider !== "custom" || requestOverrides?.temperature === undefined) return false
  if (status !== 400 && status !== 422) return false
  const detail = errorDetail.toLowerCase()
  return detail.includes("temperature") && (
    detail.includes("unsupported") ||
    detail.includes("not support") ||
    detail.includes("unknown") ||
    detail.includes("not allowed") ||
    detail.includes("only") ||
    detail.includes("invalid")
  )
}

/**
 * Field names the provider layer adds when a generic custom gateway is asked to
 * stop thinking (`reasoning: { mode: "off" }`). A gateway that does not know
 * them answers 400/422; we then re-issue the request with thinking left on so an
 * ingest still runs instead of failing outright.
 */
const REASONING_DISABLE_FIELD_HINTS = [
  "chat_template_kwargs",
  "enable_thinking",
  "reasoning_effort",
  "extra fields",
  "extra inputs",
  "extra_forbidden",
  "unknown field",
  "unexpected keyword",
  "additional propert",
  "not permitted",
]

function shouldRetryWithoutReasoningFields(
  config: LlmConfig,
  status: number,
  errorDetail: string,
  requestOverrides?: RequestOverrides,
): boolean {
  if (config.provider !== "custom") return false
  if (requestOverrides?.reasoning?.mode !== "off") return false
  if (status !== 400 && status !== 422) return false
  const detail = errorDetail.toLowerCase()
  return REASONING_DISABLE_FIELD_HINTS.some((hint) => detail.includes(hint))
}

export async function streamChat(
  config: LlmConfig,
  messages: import("./llm-providers").ChatMessage[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
  /**
   * Wire-agnostic sampling knobs. The provider's buildBody() translates
   * these into its native schema — OpenAI-style wires accept them at
   * the top level ({temperature: 0.1}), Gemini nests them under
   * generationConfig with renamed keys ({generationConfig: {temperature: 0.1}}).
   * Previously we spread them onto the body here, which broke Gemini
   * with "Unknown name 'temperature': Cannot find field." HTTP 400.
   */
  requestOverrides?: RequestOverrides,
): Promise<void> {
  const { onToken, onDone, onError } = callbacks

  // Claude Code CLI uses a subprocess transport (stdin/stdout), not
  // HTTP. Dispatch before getProviderConfig — that function throws for
  // this provider because it has no URL/headers.
  if (config.provider === "claude-code") {
    return streamViaClaudeCodeCli(
      config,
      messages,
      config.streamingEnabled === false ? bufferedStreamCallbacks(callbacks) : callbacks,
      signal,
      requestOverrides,
    )
  }

  if (config.provider === "codex-cli") {
    return streamViaCodexCli(
      config,
      messages,
      config.streamingEnabled === false ? bufferedStreamCallbacks(callbacks) : callbacks,
      signal,
      requestOverrides,
    )
  }

  const providerConfig = getProviderConfig(config)

  // Combined abort: (a) user cancel, (b) our long-horizon timeout.
  // The long timeout is a backstop for truly stuck requests; it's NOT
  // what fires when a user sees "Timeout" after 2 seconds — that is
  // almost always a fast network failure (DNS, TLS, 404, refused) that
  // WebKit surfaces as a generic "Load failed". We track whether the
  // backstop actually fired so we can tell the two apart in the error.
  const timeoutMinutes = Math.max(1, Math.min(1440, config.requestTimeoutMinutes ?? 30))
  const timeoutMs = timeoutMinutes * 60 * 1000
  let combinedSignal = signal
  let timeoutController: AbortController | undefined
  let timeoutFired = false

  if (typeof AbortSignal.timeout === "function") {
    timeoutController = new AbortController()
    const timeoutId = setTimeout(() => {
      timeoutFired = true
      timeoutController?.abort()
    }, timeoutMs)

    if (signal) {
      signal.addEventListener("abort", () => {
        clearTimeout(timeoutId)
        timeoutController?.abort()
      })
    }
    combinedSignal = timeoutController.signal
  }

  let response: Response
  try {
    const body = providerConfig.buildBody(messages, requestOverrides)
    const httpFetch = await getHttpFetch()
    response = await httpFetch(providerConfig.url, {
      method: "POST",
      headers: providerConfig.headers,
      body: JSON.stringify(body),
      signal: combinedSignal,
    })
  } catch (err) {
    if (signal?.aborted) {
      onDone()
      return
    }
    if ((err instanceof Error && err.name === "AbortError") || isRequestCancelledError(err)) {
      // Backstop timeout aborted the request (we tracked this via
      // timeoutFired); treat it as a real timeout rather than a cancel.
      if (timeoutFired) {
        onError(new Error(`Request timed out after ${Math.round(timeoutMs / 60000)} min. Try a faster model or a smaller context.`))
        return
      }
      onDone()
      return
    }
    if (isFetchNetworkError(err)) {
      if (timeoutFired) {
        onError(new Error(`Request timed out after ${Math.round(timeoutMs / 60000)} min. Try a faster model or a smaller context.`))
        return
      }
      // Fast fetch failure: DNS, TLS handshake, connection refused,
      // wrong endpoint, CORS preflight rejection, etc. All webviews
      // collapse this class of failure into an opaque error — point
      // users at the likely cause (endpoint / key / connectivity).
      onError(new Error(`Network error reaching ${providerConfig.url}. Check endpoint URL, API key, and connectivity.`))
      return
    }
    onError(err instanceof Error ? err : new Error(String(err)))
    return
  }

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`
    try {
      const body = await response.text()
      if (body) errorDetail += ` — ${body}`
    } catch {
      // ignore body read failure
    }
    if (shouldRetryWithoutTemperature(config, response.status, errorDetail, requestOverrides)) {
      const { temperature: _temperature, ...retryOverrides } = requestOverrides ?? {}
      return streamChat(config, messages, callbacks, signal, retryOverrides)
    }
    if (shouldRetryWithoutReasoningFields(config, response.status, errorDetail, requestOverrides)) {
      // The gateway rejected the "stop thinking" fields. Re-issue with thinking
      // left on: the user asked for off, but a working ingest with thinking
      // beats a hard failure, and the reasoning-only retry still covers a
      // runaway chain-of-thought.
      return streamChat(config, messages, callbacks, signal, {
        ...(requestOverrides ?? {}),
        reasoning: { mode: "auto" },
      })
    }
    if (
      response.status === 404 &&
      (config.provider === "azure" ||
        (config.provider === "custom" && isAzureOpenAiEndpoint(config.customEndpoint)))
    ) {
      onError(
        new Error(
          `${errorDetail} — Azure 404 usually means the deployment name is wrong. ` +
            `Set Model to your Azure deployment name (not the model SKU), ` +
            `and Endpoint to https://<resource>.openai.azure.com ` +
            `or .../openai/deployments/<deployment-name>.`,
        ),
      )
      return
    }
    onError(new Error(errorDetail))
    return
  }

  if (!providerConfig.streaming) {
    try {
      const payload: unknown = await response.json()
      const content = providerConfig.parseResponse(payload)
      if (!content) {
        onError(new Error("Model returned an empty non-streaming response"))
        return
      }
      onToken(content)
      onDone()
    } catch (err) {
      if (timeoutFired) {
        onError(new Error(`Request timed out after ${Math.round(timeoutMs / 60000)} min. Try a faster model or a smaller context.`))
        return
      }
      if (
        signal?.aborted ||
        (err instanceof Error && err.name === "AbortError") ||
        isRequestCancelledError(err)
      ) {
        onDone()
        return
      }
      if (isFetchNetworkError(err)) {
        onError(new Error("Connection lost while reading the complete response. Try again."))
        return
      }
      onError(err instanceof Error ? err : new Error(String(err)))
    }
    return
  }

  if (!response.body) {
    onError(new Error("Response body is null"))
    return
  }

  const reader = response.body.getReader()
  // TextDecoder keeps partial multi-byte state, so it must be scoped to this
  // response rather than shared across concurrent research requests.
  const decoder = new TextDecoder()
  let lineBuffer = ""

  // Diagnostic counters. Some OpenAI-compatible endpoints stream
  // chain-of-thought through a `reasoning_content` (DeepSeek-R1,
  // Kimi K2.x) or `reasoning` (Qwen-flavored deployments) field
  // and only put the actual answer in `delta.content` after
  // thinking ends. Misbehaving endpoints sometimes emit kilobytes
  // of reasoning and end the stream with no content at all,
  // leaving the user with a silent empty analysis. We track the
  // two channels separately so the stream-end path can tell the
  // difference between "model said nothing" and "model thought
  // out loud but never produced an answer". See reasoning-
  // detector.ts.
  let contentCharsEmitted = 0
  let reasoningCharsObserved = 0
  const recordToken = (text: string) => {
    contentCharsEmitted += text.length
    onToken(text)
  }
  const recordReasoning = (line: string) => {
    const reasoningParts = extractReasoningTextFromLine(line)
    for (const part of reasoningParts) {
      callbacks.onReasoningToken?.(part)
    }
  }
  const processRecord = (line: string): Error | null => {
    const trimmed = line.trim()
    if (!trimmed) return null

    reasoningCharsObserved += countReasoningCharsInLine(trimmed)
    recordReasoning(trimmed)
    const token = providerConfig.parseStream(trimmed)
    if (token !== null) {
      recordToken(token)
      return null
    }
    return parseEndpointErrorEnvelope(trimmed)
  }
  const stopForEndpointError = async (error: Error) => {
    // An endpoint can emit an error event without closing its SSE response.
    // Cancel the body so that the transport does not keep the connection and
    // its buffers alive after the caller has already received the failure.
    try {
      await reader.cancel()
    } catch {
      // Preserve the endpoint's actionable error if transport cleanup fails.
    }
    onError(error)
  }

  try {
    while (true) {
      const { done, value } = await reader.read()

      if (done) {
        const finalText = lineBuffer + decoder.decode()
        for (const line of splitFinalStreamRecords(finalText)) {
          const endpointError = processRecord(line)
          if (endpointError) {
            await stopForEndpointError(endpointError)
            return
          }
        }
        break
      }

      const [lines, remaining] = parseLines(decoder, value, lineBuffer)
      lineBuffer = remaining

      for (const line of lines) {
        const endpointError = processRecord(line)
        if (endpointError) {
          await stopForEndpointError(endpointError)
          return
        }
      }
    }

    // Stream ended cleanly. If the model produced thinking tokens
    // but no actual answer, surface that as a clear diagnostic
    // instead of letting the caller silently see "" (which usually
    // surfaces several layers up as "analysis not available" with
    // no clue why). Threshold guards against single-stray-byte
    // false positives from spurious empty `reasoning:""` deltas.
    const REASONING_DIAGNOSTIC_THRESHOLD = 200
    if (
      contentCharsEmitted === 0 &&
      reasoningCharsObserved >= REASONING_DIAGNOSTIC_THRESHOLD
    ) {
      onError(
        new Error(
          `Model produced ${reasoningCharsObserved.toLocaleString()} characters of reasoning / chain-of-thought, but no actual response content. ` +
          `This usually means the endpoint hit a thinking-token limit, the model didn't transition from thinking to answering, ` +
          `or the endpoint is misbehaving (the official Anthropic / OpenAI APIs don't have this issue). ` +
          `Try a shorter input, increase max_tokens, or switch to a different model in Settings.`,
        ),
      )
      return
    }

    onDone()
  } catch (err) {
    // The abort can reach us two ways: a real AbortError, or — when the
    // Tauri HTTP plugin tears down the body stream — a bare *string*
    // "Request cancelled" passed to controller.error(). The latter is not
    // an Error, so the old `err instanceof Error` guard let it fall through
    // to the generic branch and surface verbatim. Recognize both shapes.
    const isAbort =
      signal?.aborted ||
      timeoutFired ||
      (err instanceof Error && err.name === "AbortError") ||
      isRequestCancelledError(err)
    if (isAbort) {
      // Mirror the pre-fetch catch: distinguish our long-horizon backstop
      // (an actionable timeout) from a user-initiated cancel (silent).
      if (timeoutFired) {
        onError(new Error(`Request timed out after ${Math.round(timeoutMs / 60000)} min. Try a faster model or a smaller context.`))
        return
      }
      onDone()
      return
    }
    if (isFetchNetworkError(err)) {
      // Stream reader threw a network error mid-response (connection
      // dropped, server closed early, network blip). Same message
      // regardless of whether the webview is WebKit or Chromium.
      onError(new Error("Connection lost during streaming. Try again."))
      return
    }
    onError(err instanceof Error ? err : new Error(String(err)))
  } finally {
    reader.releaseLock()
  }
}

/**
 * Upper bound for the automatic reasoning-budget retry. Big enough to hold a
 * long chain-of-thought *and* its answer on any current model; small enough
 * that the retry still fits the per-request ceiling gateways commonly impose.
 */
export const REASONING_RETRY_MAX_TOKENS = 32_768

/**
 * Never retry below this. A 4x bump from a small budget can still be too tight
 * for a model that already spent thousands of tokens thinking.
 */
const REASONING_RETRY_FLOOR_TOKENS = 16_384

/**
 * Output budget to try next after a reasoning-only response, or `null` when the
 * current budget already sits at the ceiling. Returning `null` matters:
 * re-sending an identical request would fail identically, so the caller should
 * surface the original diagnostic instead of burning a round trip.
 */
function nextReasoningRetryBudget(overrides?: RequestOverrides): number | null {
  const current = overrides?.max_tokens ?? 4_096
  const bumped = Math.min(
    REASONING_RETRY_MAX_TOKENS,
    Math.max(current * 4, REASONING_RETRY_FLOOR_TOKENS),
  )
  return bumped > current ? bumped : null
}

/**
 * `streamChat` with one recovery attempt for the reasoning-only failure.
 *
 * Structured call sites (ingest analysis, long-source chunk analysis) hand the
 * model a fixed output budget. An endpoint that thinks before answering spends
 * part of that same budget on chain-of-thought, and when the thinking alone
 * fills it the stream ends on a clean stop with zero `content` — which
 * `streamChat` reports as the "produced N characters of reasoning ... but no
 * actual response content" diagnostic. Nothing about the request was invalid,
 * so the fix is to give the model room to finish thinking *and* write, rather
 * than fail the ingest and drop the page.
 *
 * The retry is taken only when the first attempt emitted no content at all, so
 * a caller can never observe duplicated content. Reasoning tokens are forwarded
 * as they arrive, so a retried attempt replays them; the structured callers
 * this exists for do not subscribe to reasoning.
 */
export async function streamChatWithReasoningRetry(
  config: LlmConfig,
  messages: import("./llm-providers").ChatMessage[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
  requestOverrides?: RequestOverrides,
): Promise<void> {
  let overrides = requestOverrides

  for (;;) {
    // Count characters, not callbacks. `streamChat` raises the reasoning-only
    // diagnostic on `contentCharsEmitted === 0`, and OpenAI-compatible
    // gateways put `content: ""` in both the role-only opening chunk and the
    // finish_reason chunk of an otherwise ordinary stream. Treating those empty
    // deltas as "the model answered" made the retry unreachable on exactly the
    // endpoints it exists for — the #743 case. Keeping the same accounting as
    // the detector guarantees the retry fires exactly when the diagnostic does.
    let contentChars = 0

    // Every path in streamChat settles through onDone or onError, so resolving
    // from the callbacks — and catching a stray rejection — cannot hang.
    const error = await new Promise<Error | undefined>((resolve) => {
      void streamChat(
        config,
        messages,
        {
          onToken: (token) => {
            contentChars += token.length
            callbacks.onToken(token)
          },
          onReasoningToken: callbacks.onReasoningToken,
          onDone: () => resolve(undefined),
          onError: (err) => resolve(err),
        },
        signal,
        overrides,
      ).catch((err: unknown) => {
        resolve(err instanceof Error ? err : new Error(String(err)))
      })
    })

    // Success and user cancellation both settle this way; neither retries.
    if (error === undefined) {
      callbacks.onDone()
      return
    }

    if (contentChars === 0 && isReasoningOnlyResponseError(error)) {
      const bumped = nextReasoningRetryBudget(overrides)
      if (bumped !== null) {
        overrides = { ...overrides, max_tokens: bumped }
        continue
      }
    }

    callbacks.onError(error)
    return
  }
}
