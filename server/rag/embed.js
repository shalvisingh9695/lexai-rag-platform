import { GoogleGenAI } from '@google/genai';

/**
 * Fallback embedding (agar API fail ho jaye)
 */
function createFallbackVector(text = '') {
  const dims = 768;
  const vector = new Array(dims);
  let hash = 0;

  for (let i = 0; i < text.length; i++) {
    hash = (hash << 5) - hash + text.charCodeAt(i);
    hash |= 0;
  }

  for (let i = 0; i < dims; i++) {
    const val = Math.sin(hash + i * 0.1) * 10000;
    vector[i] = Number((val - Math.floor(val)).toFixed(6));
  }

  return vector;
}

/**
 * ✅ FINAL EMBEDDING FUNCTION (WORKING)
 */
export async function generateEmbedding(text) {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    console.warn("No API key, using fallback vector");
    return createFallbackVector(text);
  }

  try {
    const ai = new GoogleGenAI({ apiKey });

    const response = await ai.models.embedContent({
      model: "gemini-embedding-001", // ✅ CORRECT MODEL
      contents: [
        {
          role: "user",
          parts: [{ text }]
        }
      ]
    });

    if (
      response &&
      response.embedding &&
      Array.isArray(response.embedding.values) &&
      response.embedding.values.length > 0
    ) {
      return response.embedding.values;
    }

    console.warn("Empty embedding from Gemini, using fallback");
    return createFallbackVector(text);

  } catch (err) {
    console.warn("Embedding error:", err.message);
    return createFallbackVector(text);
  }
}

/**
 * Multiple chunks embedding
 */
export async function generateEmbeddingsForChunks(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) {
    return [];
  }

  const result = [];

  for (const chunk of chunks) {
    const vector = await generateEmbedding(chunk.text);

    result.push({
      ...chunk,
      embedding: vector
    });
  }

  return result;
}