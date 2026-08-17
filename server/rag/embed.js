import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

export async function generateEmbedding(text) {
  try {
    const res = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text
    });

    return res.data[0].embedding;
  } catch (err) {
    console.error("Embedding error:", err.message);
    return [];
  }
}

export async function generateEmbeddingsForChunks(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) return [];

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