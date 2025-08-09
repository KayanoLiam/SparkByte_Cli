/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  Content,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  GenerateContentParameters,
  GenerateContentResponse,
  Part,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { toContents } from '../code_assist/converter.js';

interface OpenAICompatibleOptions {
  baseUrl: string;
  apiKey: string;
  model: string;
  httpHeaders?: Record<string, string>;
}

/**
 * Minimal OpenAI-compatible content generator using the Chat Completions API.
 * It adapts to the @google/genai request/response types used in the app.
 */
export class OpenAICompatibleContentGenerator implements ContentGenerator {
  private readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly defaultModel: string;
  private readonly headers: Record<string, string>;

  constructor(opts: OpenAICompatibleOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/$/, '');
    this.apiKey = opts.apiKey;
    this.defaultModel = opts.model;
    this.headers = {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${this.apiKey}`,
      ...(opts.httpHeaders ?? {}),
    };
  }

  async generateContentStream(
    req: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    async function* gen(self: OpenAICompatibleContentGenerator) {
      yield await self.generateContent(req, _userPromptId);
    }
    return gen(this);
  }

  async generateContent(
    req: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<GenerateContentResponse> {
    const model = req.model || this.defaultModel;
    const contents = toContents(req.contents);

    const messages = this.convertContentsToOpenAIMessages(contents);

    const body = {
      model,
      messages,
      temperature: req.config?.temperature,
      top_p: req.config?.topP,
      // Map tools if present (function calling)
      tools: this.convertTools(req.config?.tools),
      stream: false,
    } as Record<string, unknown>;

    const resp = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`OpenAI-compatible error ${resp.status}: ${text}`);
    }
    const data = await resp.json();
    return this.fromOpenAIChatCompletion(data, model);
  }

  async countTokens(_req: CountTokensParameters): Promise<CountTokensResponse> {
    // Best-effort: OpenAI compatible APIs vary; return unknown token count.
    return { totalTokens: 0 };
  }

  async embedContent(
    _req: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Not implemented for generic OpenAI compatibility in this MVP.
    return { embeddings: [] };
  }

  private convertContentsToOpenAIMessages(contents: Content[]) {
    // Map @google/genai Content[] to OpenAI messages
    // role: 'user'|'model' => 'user'|'assistant'
    return contents.map((c) => ({
      role: c.role === 'model' ? 'assistant' : 'user',
      content: this.partsToText(c.parts ?? []),
    }));
  }

  private partsToText(parts: Part[]): string {
    // Concatenate text parts; ignore images/files for MVP
    return parts
      .map((p) => (typeof p === 'string' ? p : (p as any).text ?? ''))
      .filter(Boolean)
      .join('\n');
  }

  private convertTools(_tools: GenerateContentParameters['config'] extends infer C
    ? C extends { tools?: any }
      ? C['tools']
      : undefined
    : undefined) {
    // Tool/function calling mapping omitted in MVP.
    return undefined;
  }

  private fromOpenAIChatCompletion(data: any, model: string): GenerateContentResponse {
    const text = data?.choices?.[0]?.message?.content ?? '';
    const content: Content = {
      role: 'model',
      parts: text ? [{ text }] : [],
    };
    return {
      modelVersion: model,
      candidates: [
        {
          content,
          index: 0,
        },
      ],
      usageMetadata: {
        promptTokenCount: data?.usage?.prompt_tokens,
        candidatesTokenCount: data?.usage?.completion_tokens,
        totalTokenCount: data?.usage?.total_tokens,
      },
    } as GenerateContentResponse;
  }
}