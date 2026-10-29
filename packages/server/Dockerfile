FROM node:20 AS base
RUN npm install -g pnpm@9.15.0
WORKDIR /app

# Install dependencies
FROM base AS deps
# Build tools needed for native modules (@xenova/transformers / onnxruntime)
RUN apt-get update && apt-get install -y build-essential python3 && rm -rf /var/lib/apt/lists/*
COPY package.json pnpm-workspace.yaml pnpm-lock.yaml ./
COPY packages/server/package.json ./packages/server/package.json
# node-linker=hoisted installs every package directly into node_modules (npm layout). pnpm's isolated
# layout leaves the server's own packages under packages/server/node_modules, the image copies only
# /app/node_modules, and .prisma/client lives in the hidden store, so the runner could not import
# them without linking the store into node_modules first.
RUN pnpm install --frozen-lockfile --filter @retaindb/server... --node-linker=hoisted

# Build
FROM deps AS builder
COPY packages/server ./packages/server
RUN pnpm --filter @retaindb/server run build
RUN pnpm --filter @retaindb/server exec prisma generate

# Runtime
FROM node:20-slim AS runner
# openssl for Prisma, libstdc++/libgomp for onnxruntime native bindings
RUN apt-get update && apt-get install -y openssl libstdc++6 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY --from=builder /app/packages/server/dist ./dist
COPY --from=builder /app/packages/server/prisma ./prisma
COPY --from=builder /app/node_modules ./node_modules

ENV NODE_ENV=production
ENV PORT=3000
# The hoisted install resolved by this repo's lockfile puts prisma 6.19.3 in node_modules/.bin, so
# migrations run that bundled CLI straight from PATH - nothing is fetched, and the npx path that pulled
# prisma 7.x (P1012: it rejects this repo's schema) is gone. checkpoint is prisma's telemetry host.
ENV PATH="/app/node_modules/.bin:${PATH}"
ENV CHECKPOINT_DISABLE=1

EXPOSE 3000

CMD ["sh", "-c", "prisma migrate deploy && node dist/index.js"]
