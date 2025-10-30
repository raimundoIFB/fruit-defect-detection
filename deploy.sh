#!/bin/bash

echo "🚀 Iniciando deploy para GitHub..."

cd /home/raimundo/fruit/fruit-defect-detection-master

# Verificar se é repositório git
if [ ! -d ".git" ]; then
    echo "📁 Inicializando repositório Git..."
    git init
fi

# Adicionar arquivos
echo "📦 Adicionando arquivos..."
git add .

# Verificar status
echo "📊 Status do repositório:"
git status

# Fazer commit
echo "💾 Fazendo commit..."
git commit -m "feat: $1"

# Verificar remote
if ! git remote | grep -q "origin"; then
    echo "🔗 Configurando remote..."
    git remote add origin https://github.com/raimundoIFB/fruit-defect-detection.git
fi

# Fazer push
echo "⬆️ Enviando para GitHub..."
git branch -M main
git push -u origin main

echo "✅ Deploy concluído!"