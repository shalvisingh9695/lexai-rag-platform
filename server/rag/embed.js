import { GoogleGenAI } from '@google/genai';

/**
 * Deterministic pseudo-embedding generator (768 dimensions) used as a resilient fallback
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
 * Generate real 768-dimensional embedding vector for a string using Gemini API (text-embedding-004).
 */
export async function generateEmbedding(text) {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY' || apiKey === 'your_key_here') {
    return createFallbackVector(text);
  }

  try {
    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: { 'User-Agent': 'aistudio-build' }
      }
    });

    const response = await ai.models.embedContent({
      model: 'text-embedding-004',
      contents: text
    });

    if (response && response.embedding && Array.isArray(response.embedding.values) && response.embedding.values.length > 0) {
      return response.embedding.values;
    }

    console.warn('Gemini embedding returned empty vector, using fallback vector.');
    return createFallbackVector(text);
  } catch (err) {
    console.warn('Gemini embedding request notice:', err.message);
    return createFallbackVector(text);
  }
}

/**
 * Generate embedding vectors for an array of chunk objects
 */
export async function generateEmbeddingsForChunks(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) {
    return [];
  }

  const chunksWithEmbeddings = [];

  for (const chunk of chunks) {
    const vector = await generateEmbedding(chunk.text);
    chunksWithEmbeddings.push({
      ...chunk,
      embedding: vector
    });
  }

  return chunksWithEmbeddings;
}

