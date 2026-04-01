# saas-churn-prediction-shap

## Contexte Métier
Dans le secteur des télécommunications, acquérir un nouveau client coûte beaucoup plus cher que de retenir un client existant. L'objectif de ce projet est de construire un modèle de Machine Learning capable de prédire quels clients sont susceptibles de résilier leur abonnement (Churn) afin de déclencher des campagnes de rétention ciblées.

## Démarche Technique
Face à un jeu de données fortement déséquilibré (73% de clients fidèles vs 27% de départs), l'enjeu principal était de maximiser la détection des résiliations sans exploser le budget marketing avec de fausses alertes.

* **Prétraitement et Feature Engineering :** Encodage des variables catégorielles (One-Hot), mise à l'échelle (MinMaxScaler) et création de nouvelles variables (évolution du prix, score d'engagement).
* **Benchmarking de Modèles :** Évaluation de multiples algorithmes via un pipeline automatisé (Logistic Regression, Random Forest, Gradient Boosting, SVM, XGBoost).
* **Optimisation des Hyperparamètres :** Fine-Tuning du meilleur modèle (XGBoost) via RandomizedSearchCV avec une validation croisée robuste (5 folds).
* **Choix de la Métrique :** Abandon de l'Accuracy globale au profit d'une optimisation basée sur le F1-Score et le Rappel (Recall) de la classe minoritaire.
* **Ajustement des Seuils :** Manipulation des probabilités de prédiction (predict_proba) pour créer des segments de risque personnalisés (> 70% de risque).

## Résultats
Le modèle final (XGBoost Fine-Tuné) a été configuré pour privilégier la détection stricte des départs (minimisation des Faux Négatifs).

* **Rappel (Recall) sur le Churn : 81%** (Le modèle identifie avec succès plus de 8 clients sur 10 prêts à partir).
* **F1-Score : 0.62** (Excellent compromis pour ce type de données déséquilibrées).
* **Accuracy Globale : 74%**
* **Matrice de Confusion sur le set de test (1407 clients) :**
  * Vrais Négatifs (Restent) : 732
  * Faux Positifs (Fausses alertes) : 301
  * Faux Négatifs (Départs ratés) : 71
  * **Vrais Positifs (Départs anticipés) : 303**

## Stack Technique
* **Langage :** Python 3
* **Librairies ML :** Scikit-Learn, XGBoost
* **Manipulation de données :** Pandas, NumPy
* **Évaluation :** GridSearchCV, RandomizedSearchCV, Classification Report
