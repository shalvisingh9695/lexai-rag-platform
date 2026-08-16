import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';

import documentRoutes from './routes/documentRoutes.js';
import aiRoutes from './routes/aiRoutes.js';
import chatRoutes from './routes/chatRoutes.js';
import searchRoutes from './routes/searchRoutes.js';
import askRoutes from './routes/askRoutes.js';

import { dbConnect, getInMemoryStore } from './utils/dbConnect.js';
import { ensureDocumentIndexed } from './rag/ragService.js';

const app = express();

// ✅ SIMPLE & CORRECT PORT (Render compatible)
const PORT = process.env.PORT || 10000;

// ✅ Middlewares
app.use(cors());
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// ✅ Routes
app.use('/api/documents', documentRoutes);
app.use('/api/search', searchRoutes);
app.use('/api/ask', askRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/summary', aiRoutes);
app.use('/api/risk', aiRoutes);
app.use('/api/compare', aiRoutes);
app.use('/api/chat', chatRoutes);

// ✅ Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'LexAI Backend',
    port: PORT,
    timestamp: new Date().toISOString()
  });
});

// ✅ Serve frontend (Vite build)
const distPath = path.join(process.cwd(), 'dist');
if (fs.existsSync(distPath)) {
  app.use(express.static(distPath));
  app.get('*', (req, res) => {
    res.sendFile(path.join(distPath, 'index.html'));
  });
}

// ✅ Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Server running on ${PORT}`);

  // ✅ DB connect
  dbConnect()
    .then(() => {
      console.log('✅ MongoDB Connected');

      try {
        const store = getInMemoryStore();
        if (store && store.documents) {
          for (const doc of store.documents) {
            ensureDocumentIndexed(doc._id || doc.id).catch(() => {});
          }
        }
      } catch (e) {
        console.warn('RAG pre-seed notice:', e.message);
      }
    })
    .catch(err => {
      console.error('❌ MongoDB Error:', err.message);
    });
});