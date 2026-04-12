#!/bin/bash

# Script de compilation pour le papier de recherche DPI
# Dépendances : pandoc, pdfunite, citeproc

echo "🚀 Début de la compilation du papier DPI..."

# 1. Compilation de l'Abstract seul (pour qu'il apparaisse avant la TOC)
pandoc metadata.yaml sections/01_abstract.md -o abstract.pdf

# 2. Compilation du corps du texte (Introduction -> Conclusion)
# --toc-depth=3 : Limite la Table des Matières à 3 niveaux pour la clarté
# --citeproc : Résout les citations bibliographiques
pandoc metadata.yaml \
    sections/02_introduction.md \
    sections/02_genesis.md \
    sections/03_methodology.md \
    sections/04_results.md \
    sections/04_baseline_comparison.md \
    sections/04_detailed_metrics.md \
    sections/04_marathon_results.md \
    sections/04_scaling_results.md \
    sections/04_billions_scaling.md \
    sections/04_heterogeneity_results.md \
    sections/04_ablation_results.md \
    sections/04_sensitivity_results.md \
    sections/04_discussion.md \
    sections/05_conclusion.md \
    sections/06_references.md \
    --toc \
    --toc-depth=3 \
    --mathjax \
    --citeproc \
    --bibliography=references.bib \
    -o body.pdf

# 3. Fusion de l'Abstract et du Corps du texte
if [ -f abstract.pdf ] && [ -f body.pdf ]; then
    pdfunite abstract.pdf body.pdf DPI_Research_Paper.pdf
    echo "✅ PDF FINAL GÉNÉRÉ : DPI_Research_Paper.pdf"
    
    # Nettoyage des fichiers temporaires
    rm abstract.pdf body.pdf
else
    echo "❌ Erreur lors de la génération des PDF intermédiaires."
    exit 1
fi
