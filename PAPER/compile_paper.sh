#!/bin/bash

# Script de compilation pour le papier de recherche DPI
# Dépendances : pandoc, pdfunite, citeproc

echo "🚀 Début de la compilation du papier DPI..."

# 1. Compilation de l'Abstract avec Titre et Auteurs
# Cette partie génère la première page avec le titre, les auteurs et le résumé.
pandoc metadata.yaml sections/01_abstract.md -o abstract.pdf

# 2. Création d'un metadata temporaire sans titre pour le corps du texte
# Cela évite que Pandoc ne répète le titre au début du corps (page 2).
grep -v "title:" metadata.yaml > metadata_no_title.yaml

# 3. Compilation du corps du texte (Introduction -> Conclusion)
# --toc-depth=2 : Limite la Table des Matières à 2 niveaux (Sections et Sous-sections)
# --citeproc : Résout les citations
pandoc metadata_no_title.yaml \
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
    sections/04_investigations_intro.md \
    sections/04_sampling_density.md \
    sections/04_ablation_results.md \
    sections/04_sensitivity_results.md \
    sections/04_discussion.md \
    sections/05_conclusion.md \
    sections/07_appendix.md \
    sections/08_appendix_holy_grail.md \
    sections/09_appendix_conductivity.md \
    sections/06_references.md \
    \
    --toc \
    --toc-depth=2 \
    --mathjax \
    --citeproc \
    --bibliography=references.bib \
    -o body.pdf

# 4. Fusion de l'Abstract et du Corps du texte
if [ -f abstract.pdf ] && [ -f body.pdf ]; then
    pdfunite abstract.pdf body.pdf DPI_Research_Paper.pdf
    echo "✅ PDF FINAL GÉNÉRÉ : DPI_Research_Paper.pdf"
    
    # Nettoyage des fichiers temporaires
    rm abstract.pdf body.pdf metadata_no_title.yaml
else
    echo "❌ Erreur lors de la génération des PDF intermédiaires."
    exit 1
fi
