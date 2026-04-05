---
project: 'Edram — Sephora Customer Segmentation ML'
author: 'Bob (SM) pour Milan'
date: '2026-04-02'
version: '1.0'
type: 'Epic — Refonte majeure'
impacts: 'Remplace E3 (Feature Engineering & Preprocessing) + impacte E4 (Clustering)'
---

# EPIC R1 — Refonte Feature Engineering, Explicabilité & MLflow

> **Epic Goal:** Réduire et restructurer les variables en 6 catégories marketing-actionnables, supprimer la PCA au profit de l'explicabilité, ajouter un cercle des corrélations pour validation humaine, et intégrer MLflow pour le tracking expérimental — le tout dans un notebook refondu.

**Value Statement:**
*En tant que Marketing Manager, je veux des clusters construits sur des variables que je comprends et que je peux activer (Profil, Valeur, Affinité Produit, Comportement, Canal, Dates), sans boîte noire PCA, avec un suivi MLflow de chaque expérience de clustering.*

**Contexte de la refonte:**
Les changements suivants ont été demandés par Milan :
1. Classifier chaque variable dans une catégorie marketing (Profil / Valeur / Affinité Produit / Comportement / Canal / Dates)
2. Supprimer les variables inutiles au ciblage marketing (volume pur, localisation trop fine, passé lointain)
3. Ajouter un cercle des corrélations pour décision humaine de suppression finale
4. Supprimer la PCA — conserver les features brutes pour l'explicabilité
5. Refondre le notebook avec intégration MLflow

---

## Definition of Done (Epic-level)

- [ ] Toutes les variables classifiées dans une des 6 catégories ou explicitement supprimées
- [ ] Aucune variable "volume pur", localisation trop granulaire, ou passé lointain inutile ne subsiste
- [ ] Le cercle des corrélations est affiché et exploitable
- [ ] Aucune PCA n'est utilisée dans le pipeline de clustering
- [ ] MLflow tracke chaque run de clustering (params, métriques, artefacts)
- [ ] Le notebook 02_clustering.ipynb est refondu et cohérent de bout en bout

---

## US-R1.1 — Classification des variables en 6 catégories marketing

**As a** Marketing Manager,
**I want to** voir chaque variable classifiée dans une catégorie métier claire,
**so that** je sais exactement ce que chaque feature mesure et comment elle sert au ciblage.

**Priorité:** P0 — fondation de toute la refonte

**Acceptance Criteria:**
- [ ] Chaque variable de `customers_features.csv` est classée dans exactement une des catégories suivantes :

| Catégorie | Description | Variables attendues |
|---|---|---|
| **Profil** | Qui est le client (socio-démo) | `age`, `gender`, `country`, `loyalty_numeric` |
| **Valeur** | Combien le client rapporte | `recency_days`, `frequency`, `monetary_total`, `monetary_avg`, `avg_basket_size_eur`, `discount_rate` |
| **Affinité Produit** | Quoi le client achète | `axe_make_up_ratio`, `axe_skincare_ratio`, `axe_fragrance_ratio`, `axe_haircare_ratio`, `axe_others_ratio`, `dominant_axe`, `market_selective_ratio`, `market_exclusive_ratio`, `market_sephora_ratio`, `market_others_ratio`, `dominant_market`, `axis_diversity` |
| **Comportement** | Comment le client achète | `avg_units_per_basket`, `nb_unique_brands`, `nb_unique_stores` |
| **Canal** | Où le client achète | `store_ratio`, `estore_ratio`, `click_collect_ratio`, `dominant_channel` |
| **Dates** | Quand / ancienneté | `subscription_tenure_days` |

- [ ] Un dictionnaire `FEATURE_CATEGORIES` est créé dans `src/config.py` avec la structure :
  ```python
  FEATURE_CATEGORIES = {
      "profil": [...],
      "valeur": [...],
      "affinite_produit": [...],
      "comportement": [...],
      "canal": [...],
      "dates": [...],
  }
  ```
- [ ] Les listes `FEATURES_CONTINUOUS`, `FEATURES_ONEHOT` sont mises à jour en cohérence
- [ ] Un tableau Markdown récapitulatif est affiché dans le notebook

**Technical Notes:**
- Le dictionnaire `FEATURE_CATEGORIES` servira à colorer le cercle des corrélations (US-R1.3)
- Les catégories `dominant_*` sont catégorielles → One-Hot ; les `*_ratio` sont continues → scaler

---

## US-R1.2 — Suppression des variables non-actionnables

**As a** Marketing Manager,
**I want to** retirer toutes les variables qui ne permettent pas de prendre une décision de ciblage,
**so that** le modèle n'est pas pollué par du bruit et les clusters sont interprétables.

**Priorité:** P0

**Acceptance Criteria:**
- [ ] Les variables suivantes sont **supprimées** (ajoutées à `FEATURES_DROP`) avec la justification :

| Variable supprimée | Raison |
|---|---|
| `total_quantity` | Volume pur — conséquence de fidélité, pas un levier de ciblage. `avg_units_per_basket` est gardé. |
| `total_lines` | Volume pur — même raison que `total_quantity`. |
| `customer_city` | Trop granulaire (~12K modalités). Inutilisable en clustering. Pas de transformation en zone urbaine/rurale prévue. |
| `first_purchase_amount` | Passé lointain — ne reflète plus le comportement actuel. |
| `channel_recruitment` | Passé lointain — `dominant_channel` le remplace pour le comportement actuel. |
| `first_purchase_axe` | Passé lointain — `dominant_axe` le remplace. |
| `age_category` | Redondant avec `age` (qui est continu et plus riche). |
| `age_generation` | Redondant avec `age`. |
| `total_discount_eur` | Redondant — `discount_rate` capture la même info en ratio normalisé. |
| `cc_transactions` | Redondant — `click_collect_ratio` existe déjà en ratio. |
| `total_transactions` | Déjà supprimé (duplicate `frequency`). Confirmer. |
| `total_sales_eur` | Déjà supprimé (duplicate `monetary_total`). Confirmer. |
| `avg_sales_eur` | Déjà supprimé (duplicate `monetary_avg`). Confirmer. |
| `has_age_info` | Indicateur de missing — bruit potentiel, pas actionnable. |
| `has_first_purchase_info` | Indicateur de missing — idem. |
| `customer_city_freq` | Suppression car `customer_city` supprimée (pas de frequency encoding). |

- [ ] `FEATURES_DROP` dans `src/config.py` est mis à jour
- [ ] `FEATURES_FREQUENCY` est vidé (plus de frequency encoding nécessaire)
- [ ] `FEATURES_ONEHOT` est réduit : retirer `channel_recruitment`, `age_category`, `age_generation`, `first_purchase_axe`, `customer_city`
- [ ] La liste finale des **One-Hot** se limite à : `gender`, `dominant_channel`, `dominant_axe`, `dominant_market`, `country`  
  → Cela génère environ **18 dummy columns** au lieu de 37
- [ ] Le preprocessing pipeline `preprocess_for_clustering()` est mis à jour en conséquence
- [ ] Shape finale attendue : environ **(64469, ~40-45 features)** au lieu de 70

**Règle de décision :** Si c'est évident qu'il faut supprimer → supprimer. Si c'est discutable → garder et la mettre dans une catégorie "autre". Dans le doute, on garde.

**Technical Notes:**
- `nb_unique_brands` et `nb_unique_stores` : gardés (Comportement). Pas des volumes purs, ils mesurent la diversité.
- `market_sephora_ratio`, `market_others_ratio` : gardées (Affinité Produit). Elles complètent le market mix.
- `axis_diversity` : gardée (Affinité Produit). C'est un indicateur de polyvalence, utile pour cibler.

---

## US-R1.3 — Cercle des corrélations (décision humaine)

**As a** Data Analyst,
**I want to** afficher un cercle des corrélations sur les features scalées,
**so that** Milan puisse visuellement identifier les variables restantes redondantes et décider lesquelles supprimer.

**Priorité:** P1 — dépend de R1.1 et R1.2

**Acceptance Criteria:**
- [ ] Le cercle des corrélations est calculé via PCA à 2 composantes **uniquement pour la visualisation** (⚠️ PCA n'est PAS utilisée pour le clustering, seulement pour projeter le cercle)
- [ ] Chaque flèche du cercle représente une variable, colorée par sa catégorie (`FEATURE_CATEGORIES`)
- [ ] Les variables proches (angle < 10°) sont identifiées et listées automatiquement dans un print
- [ ] Le cercle est affiché avec `matplotlib` et sauvegardé dans `figures/`
- [ ] Une légende montre les 6 couleurs = 6 catégories
- [ ] Milan décidera manuellement quelles variables supprimer après inspection du graphe
- [ ] Fonction `plot_correlation_circle()` ajoutée dans `src/visualization.py`

**Technical Notes:**
- Calcul : PCA(n_components=2) sur `X_scaled`, puis projection des axes originaux (colonnes de la matrice de loadings = `pca.components_.T`)
- Colorer les flèches par catégorie (utiliser `FEATURE_CATEGORIES` de config)
- Tracer le cercle unité
- Annoter chaque flèche avec le nom de la variable
- Si trop de variables se chevauchent, ajuster `fontsize` ou utiliser `adjustText`

---

## US-R1.4 — Suppression de la PCA du pipeline de clustering

**As a** Marketing Manager,
**I want to** que le clustering opère directement sur les features scalées (sans réduction PCA),
**so that** chaque cluster puisse être expliqué variable par variable, sans passer par des composantes abstraites.

**Priorité:** P0

**Acceptance Criteria:**
- [ ] La fonction `apply_pca()` dans `src/preprocessing.py` n'est plus appelée dans le pipeline de clustering
- [ ] La décision PCA dans le notebook (US-3.4) est remplacée par une cellule Markdown expliquant le choix de ne PAS utiliser la PCA pour l'explicabilité
- [ ] Le clustering (`KMeans`, `Hierarchical`, etc.) opère sur `X_scaled` directement
- [ ] Les fonctions `apply_pca()` et `apply_umap()` restent dans le code (ne pas casser l'existant) mais ne sont plus dans le flow principal
- [ ] Le notebook ne contient plus de cellule exécutant `apply_pca()` pour le clustering
- [ ] La config `CLUSTERING_FEATURES` est mise à jour pour refléter les features scalées réelles (pas des composantes PC1..PCn)

**Technical Notes:**
- Après suppression des features (R1.2) et scaling, on devrait avoir ~40-45 features
- C'est gérable par KMeans sans PCA (la "malédiction de la dimensionnalité" est significative au-delà de ~100 features sur ce volume de données)
- La PCA à 2 composantes est TOUJOURS utilisée pour le cercle des corrélations (R1.3) — c'est un outil de visualisation, pas de modélisation

---

## US-R1.5 — Intégration MLflow pour le tracking expérimental

**As a** Data Scientist,
**I want to** tracker chaque run de clustering avec MLflow,
**so that** chaque expérience (params, métriques, artefacts) est reproductible et comparable.

**Priorité:** P1

**Acceptance Criteria:**
- [ ] `mlflow` installé et importé dans le notebook
- [ ] Un experiment MLflow nommé `"sephora-customer-segmentation"` est créé
- [ ] Chaque run de clustering (KMeans, Hierarchical, etc.) logge :
  - **Params** : `k`, `algorithm`, `n_features`, `scaling_method`, `random_state`
  - **Metrics** : `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, `min_cluster_pct`, `max_cluster_pct`
  - **Artefacts** : le CSV des labels, les figures (elbow, silhouette, distributions)
  - **Tags** : `dataset_version`, `preprocessing_version`
- [ ] Le scan de K (k=2..30) est loggé comme un parent run avec des nested runs par valeur de k
- [ ] Les résultats sont consultables via `mlflow ui` (commande documentée dans le notebook)
- [ ] Une fonction utilitaire `log_clustering_run()` est créée dans `src/clustering.py`

**Technical Notes:**
- Backend de tracking : local (`mlruns/` dans le repo)
- Ajouter `mlruns/` au `.gitignore`
- Pattern recommandé :
  ```python
  import mlflow
  mlflow.set_experiment("sephora-customer-segmentation")
  with mlflow.start_run(run_name="kmeans-k-scan"):
      for k in K_RANGE:
          with mlflow.start_run(run_name=f"kmeans-k{k}", nested=True):
              mlflow.log_param("k", k)
              mlflow.log_metric("silhouette", score)
              ...
  ```

---

## US-R1.6 — Refonte du notebook 02_clustering.ipynb

**As a** Data Analyst,
**I want to** un notebook refondu, propre et séquentiel,
**so that** le pipeline complet soit lisible, reproductible, et exploitable pour la soutenance.

**Priorité:** P1 — dépend de toutes les stories précédentes

**Acceptance Criteria:**
- [ ] Le notebook suit cette structure de sections :

| # | Section | Contenu |
|---|---|---|
| 1 | **Setup & Imports** | Imports, MLflow init, chargement des données |
| 2 | **Feature Audit** | Tableau de classification des variables par catégorie (R1.1) |
| 3 | **Feature Selection** | Application des suppressions (R1.2), justification Markdown |
| 4 | **Preprocessing** | Imputation, encoding, scaling (pas de PCA) (R1.4) |
| 5 | **Cercle des corrélations** | Graphe + décision humaine (R1.3) |
| 6 | **Clustering — K-Scan** | KMeans k=2..30 avec MLflow tracking (R1.5) |
| 7 | **Clustering — Modèle Final** | K choisi, labels assignés |
| 8 | **Profiling des segments** | Moyennes par cluster, heatmap |
| 9 | **Export** | CSV final, MLflow artifacts |

- [ ] Chaque section commence par une cellule Markdown expliquant l'objectif
- [ ] Les cellules de code sont courtes (< 30 lignes) et commentées
- [ ] Toutes les figures sont sauvegardées dans `figures/`
- [ ] Le notebook s'exécute de haut en bas sans erreur (`Restart & Run All`)

---

## Séquence d'implémentation recommandée

```
R1.1 (Classification) ──→ R1.2 (Suppression) ──→ R1.4 (Suppr. PCA)
                                                        │
                                                        ▼
                                                  R1.3 (Cercle corrélations)
                                                        │
                                                        ▼
                                                  R1.5 (MLflow)
                                                        │
                                                        ▼
                                                  R1.6 (Refonte notebook)
```

**Estimation de complexité :**

| Story | Complexité | Points |
|---|---|---|
| R1.1 | S | 2 |
| R1.2 | M | 5 |
| R1.3 | M | 5 |
| R1.4 | S | 2 |
| R1.5 | M | 5 |
| R1.6 | L | 8 |
| **Total** | | **27** |

---

## Risques identifiés

| Risque | Impact | Mitigation |
|---|---|---|
| Trop de variables supprimées → clusters pauvres | Fort | Le cercle des corrélations (R1.3) sert de filet de sécurité — Milan décide au final |
| MLflow alourdit le notebook pour un débutant | Moyen | Encapsuler dans `log_clustering_run()` — une seule ligne d'appel |
| Sans PCA, KMeans sur ~40 features peut être lent | Faible | 64K lignes × 40 features = rapide sur sklearn, pas besoin de GPU |
| Le one-hot sur `dominant_*` crée quand même 18 dummies | Moyen | Acceptable — on passe de 37 à 18. Si Milan veut réduire encore, on peut passer en ordinal |
