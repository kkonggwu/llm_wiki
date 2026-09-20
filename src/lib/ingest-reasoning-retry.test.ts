/**
 * End-to-end proof for #743.
 *
 * Only the HTTP transport is mocked (with SSE bodies), so the real
 * `streamChat`, the real reasoning-only detector and the real
 * `streamChatWithReasoningRetry` wrapper all run. The analysis attempt is
 * scripted to fail exactly the way the issue reports — chain-of-thought only,
 * bracketed by the empty `content` deltas that OpenAI-compatible gateways
 * always send — and the test asserts the page is still written.
 *
 * Before the empty-delta fix in `streamChatWithReasoningRetry`, the analysis
 * step consumed its whole budget on CoT, the retry never fired, and the source
 * was dropped with "Analysis failed: ... no actual response content".
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest"
import {
  realFs,
  createTempProject,
  writeFileRaw,
  readFileRaw,
  fileExists,
} from "@/test-helpers/fs-temp"

vi.mock("@/commands/fs", () => realFs)

/** Request bodies in call order, plus the scripted SSE body per call. */
const requestBodies: string[] = []
let scriptedResponses: string[] = []

const mockHttpFetch = vi.fn(async (_url: string, opts?: RequestInit) => {
  requestBodies.push(String(opts?.body ?? ""))
  const payload = scriptedResponses[requestBodies.length - 1] ?? ""
  return new Response(payload, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  })
})

vi.mock("./tauri-fetch", async () => {
  const actual = await vi.importActual<typeof import("./tauri-fetch")>("./tauri-fetch")
  return { ...actual, getHttpFetch: () => Promise.resolve(mockHttpFetch) }
})

import { autoIngest } from "./ingest"
import { useWikiStore } from "@/stores/wiki-store"
import { useReviewStore } from "@/stores/review-store"
import { useActivityStore } from "@/stores/activity-store"
import { useChatStore } from "@/stores/chat-store"

const ANALYSIS_MARKER = "ANALYSIS-FROM-RETRY-9f3a"

function sse(...records: string[]): string {
  return [...records, "data: [DONE]"].join("\n\n")
}

function contentRecord(content: string): string {
  return `data: ${JSON.stringify({ choices: [{ delta: { content } }] })}`
}

function reasoningRecord(reasoning: string): string {
  return `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: reasoning } }] })}`
}

/** Opening chunk of every OpenAI-compatible stream: role only, empty content. */
function roleOnlyRecord(): string {
  return `data: ${JSON.stringify({ choices: [{ delta: { role: "assistant", content: "" } }] })}`
}

/** Closing chunk of such a stream: empty content plus finish_reason. */
function finishRecord(): string {
  return `data: ${JSON.stringify({
    choices: [{ delta: { content: "" }, finish_reason: "stop" }],
  })}`
}

function gatewayReasoningOnlyStream(): string {
  return sse(roleOnlyRecord(), reasoningRecord("t".repeat(300)), finishRecord())
}

function generationStream(): string {
  return sse(contentRecord([
    "---FILE: wiki/concepts/retry-proof.md---",
    "---",
    "type: concept",
    "title: Retry Proof",
    "sources: [reasoning-chunk.md]",
    "tags: []",
    "related: []",
    "---",
    "",
    "# Retry Proof",
    "",
    "PAGE-WRITTEN-AFTER-RETRY",
    "---END FILE---",
  ].join("\n")))
}

let ctx: { path: string; cleanup: () => Promise<void> } | undefined

beforeEach(async () => {
  requestBodies.length = 0
  scriptedResponses = []
  useReviewStore.setState({ items: [] })
  useActivityStore.setState({ items: [] })
  useChatStore.setState({
    conversations: [],
    messages: [],
    activeConversationId: null,
    mode: "chat",
    ingestSource: null,
    isStreaming: false,
    streamingContent: "",
  })

  ctx = await createTempProject("ingest-743-retry")
  const projectPath = ctx.path
  await writeFileRaw(`${projectPath}/schema.md`, "")
  await writeFileRaw(`${projectPath}/purpose.md`, "")
  await writeFileRaw(`${projectPath}/wiki/index.md`, "# Index\n")
  await writeFileRaw(`${projectPath}/wiki/overview.md`, "# Overview\n")
  await writeFileRaw(
    `${projectPath}/raw/sources/reasoning-chunk.md`,
    "# Thinking model source\n\nSome content to analyse.\n",
  )

  useWikiStore.setState({
    project: {
      name: "t",
      path: projectPath,
      createdAt: 0,
      purposeText: "",
      fileTree: [],
    } as unknown as ReturnType<typeof useWikiStore.getState>["project"],
  })
  // Mirrors the #743 setup: an OpenAI-compatible gateway in front of a
  // thinking model, with thinking explicitly requested for ingest.
  useWikiStore.getState().setLlmConfig({
    provider: "custom",
    apiKey: "test-key",
    model: "thinking-model",
    ollamaUrl: "",
    customEndpoint: "http://127.0.0.1:13305/v1",
    apiMode: "chat_completions",
    maxContextSize: 128000,
    ingestReasoning: { mode: "high" },
  })
})

afterEach(async () => {
  if (ctx) {
    await ctx.cleanup()
    ctx = undefined
  }
})

describe("ingest recovers from a reasoning-only analysis (#743)", () => {
  it("writes the page after retrying the analysis with a larger budget", async () => {
    const projectPath = ctx!.path

    scriptedResponses = [
      gatewayReasoningOnlyStream(),          // 1: analysis attempt, CoT only
      sse(contentRecord(ANALYSIS_MARKER)),   // 2: analysis retry, real answer
      generationStream(),                    // 3: generation
    ]

    await autoIngest(
      projectPath,
      `${projectPath}/raw/sources/reasoning-chunk.md`,
      useWikiStore.getState().llmConfig,
    )

    // The retry happened: three wire requests, the second one with a much
    // larger budget than the 8192 the 128k-context analysis pass starts with.
    expect(requestBodies).toHaveLength(3)
    expect(JSON.parse(requestBodies[0]).max_tokens).toBe(8_192)
    expect(JSON.parse(requestBodies[1]).max_tokens).toBe(32_768)
    // `ingestReasoning: { mode: "high" }` is not representable on a generic
    // custom gateway, so it normalizes to `auto`: no reasoning parameter is
    // sent at all and the model thinks freely. That is exactly the #743 setup,
    // and why the empty-delta guard in the wrapper mattered so much. (Selecting
    // `off` now also works on these gateways — see the provider/retry tests.)
    expect(JSON.parse(requestBodies[1]).thinking).toBeUndefined()
    expect(JSON.parse(requestBodies[1]).reasoning_effort).toBeUndefined()

    // The retry's answer is what reached generation — not the empty first pass.
    expect(requestBodies[2]).toContain(ANALYSIS_MARKER)

    // The page survived instead of being dropped with "Analysis failed".
    const pagePath = `${projectPath}/wiki/concepts/retry-proof.md`
    expect(await fileExists(pagePath)).toBe(true)
    const page = await readFileRaw(pagePath)
    expect(page).toContain("PAGE-WRITTEN-AFTER-RETRY")
    expect(page).toContain("title: Retry Proof")

    // No activity was left in the failed state by the recovered attempt.
    expect(useActivityStore.getState().items.filter((i) => i.status === "error")).toHaveLength(0)
  })

  it("writes the page when the generation pass is the one that only thought", async () => {
    const projectPath = ctx!.path

    scriptedResponses = [
      sse(contentRecord(ANALYSIS_MARKER)),   // 1: analysis
      gatewayReasoningOnlyStream(),          // 2: generation attempt, CoT only
      generationStream(),                    // 3: generation retry
    ]

    await autoIngest(
      projectPath,
      `${projectPath}/raw/sources/reasoning-chunk.md`,
      useWikiStore.getState().llmConfig,
    )

    // Generation starts from 16 384 at a 128k context, so this pins that the
    // page-producing step recovers the same way the analysis step does.
    expect(requestBodies).toHaveLength(3)
    expect(JSON.parse(requestBodies[1]).max_tokens).toBe(16_384)
    expect(JSON.parse(requestBodies[2]).max_tokens).toBe(32_768)

    const pagePath = `${projectPath}/wiki/concepts/retry-proof.md`
    expect(await fileExists(pagePath)).toBe(true)
    expect(await readFileRaw(pagePath)).toContain("PAGE-WRITTEN-AFTER-RETRY")
    expect(useActivityStore.getState().items.filter((i) => i.status === "error")).toHaveLength(0)
  })
})
