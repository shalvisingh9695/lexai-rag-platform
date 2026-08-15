/**
 * Vector Storage & Similarity Search Engine
 * Stores chunk embeddings in-memory and retrieves top-K relevant chunks using Cosine Similarity.
 */

// In-memory vector store: Map<documentId, Array<{ id, text, embedding }>>
const vectorStore = new Map();

/**
 * Calculate Cosine Similarity between two numerical vectors
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) {
    return 0;
  }

  // Support vectors of differing lengths gracefully by padding or using common length
  const minLength = Math.min(vecA.length, vecB.length);
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < minLength; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Save chunks and embeddings for a specific document
 */
export function storeDocumentChunks(documentId, chunksWithEmbeddings) {
  if (!documentId) return;
  vectorStore.set(String(documentId), chunksWithEmbeddings || []);
}

/**
 * Check if a document is present in RAG vector store
 */
export function hasDocument(documentId) {
  return vectorStore.has(String(documentId));
}

/**
 * Get all chunks for a document
 */
export function getDocumentChunks(documentId) {
  return vectorStore.get(String(documentId)) || [];
}

/**
 * Remove document chunks from vector store
 */
export function deleteDocumentChunks(documentId) {
  vectorStore.delete(String(documentId));
}

/**
 * Retrieve top-K relevant chunks for a given question embedding
 */
export function searchRelevantChunks(documentId, questionEmbedding, topK = 3) {
  const chunks = vectorStore.get(String(documentId));

  // If specific document not found or empty, check all stored chunks or return empty
  if (!chunks || chunks.length === 0) {
    // Search across all stored documents if specific ID is not found
    const allChunks = [];
    for (const [docId, docChunks] of vectorStore.entries()) {
      allChunks.push(...docChunks.map(c => ({ ...c, documentId: docId })));
    }

    if (allChunks.length === 0) return [];

    const scored = allChunks.map(chunk => ({
      ...chunk,
      similarityScore: cosineSimilarity(questionEmbedding, chunk.embedding)
    }));

    scored.sort((a, b) => b.similarityScore - a.similarityScore);
    return scored.slice(0, topK);
  }

  const scoredChunks = chunks.map(chunk => ({
    ...chunk,
    similarityScore: cosineSimilarity(questionEmbedding, chunk.embedding)
  }));

  // Sort descending by similarity score
  scoredChunks.sort((a, b) => b.similarityScore - a.similarityScore);

  return scoredChunks.slice(0, topK);
}
