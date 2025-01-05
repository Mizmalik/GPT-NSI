# Projet GPT NSI

Le projet **GPT NSI** est une application web interactive basée sur les **LLM** (Large Language Models). Elle utilise un dataset inspiré de **Homer Simpson** pour générer des réponses humoristiques et contextuelles. Le projet vise à offrir une expérience utilisateur immersive en incarnant l'esprit comique et décalé de ce personnage emblématique.

---

## Objectifs du projet

- **Créer une IA thématique** : Générer des réponses inspirées de la personnalité et de l'humour d'Homer Simpson.
- **Interface web intuitive** : Offrir une plateforme simple d'utilisation pour interagir avec l'IA.
- **Utilisation pédagogique et ludique** : Explorer les technologies modernes d'IA tout en s'amusant avec un style inspiré des *Simpsons*.

---

## Fonctionnalités principales

1. **Chat interactif** : Permet de poser des questions ou d'entamer des discussions avec une IA adoptant le style comique d'Homer Simpson.
2. **Réponses contextuelles et humoristiques** : Les réponses sont enrichies de blagues, d'expressions exagérées et d'anecdotes rappelant la série.
3. **Interface utilisateur fluide** : Intégration d'une interface graphique intuitive permettant une expérience en temps réel.

---

## Technologies utilisées

- **Modèle GPT** : Génère des réponses cohérentes et humoristiques basées sur des dialogues d'Homer Simpson.
- **Dataset thématique** : Constitué des citations, répliques et comportements du personnage, tirés de la série *Les Simpsons*.
- **Frontend** : Créé avec **HTML**, **CSS**, et **JavaScript** pour une interface dynamique.
- **Backend Flask** : Fournit une API qui traite les requêtes utilisateur et génère les réponses avec le modèle GPT.

---

## Installation et démarrage

### Prérequis

1. **Python** (version >= 3.8)
2. **Flask** et dépendances associées (voir `requirements.txt`)
3. Navigateur web pour accéder à l'interface utilisateur.

### Étapes

1. **Installer les dépendances Python** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Lancer le serveur Flask** :
   Depuis le dossier racine du projet, exécutez :
   ```bash
   python generate.py
   ```

3. **Accéder à l'interface utilisateur** :
   Ouvrez le fichier `index.html` situé dans le dossier `website` dans votre navigateur ou configurez un serveur local pour héberger l'interface.

---

## Aperçu rapide

- **Frontend** : Une interface simple avec un champ de saisie et un bouton d'envoi.
- **Backend** : Reçoit les requêtes de l'utilisateur, génère des réponses via le modèle et renvoie les résultats en JSON.

---

N'hésitez pas à explorer, discuter et vous amuser avec ce projet basé sur le style unique d'Homer Simpson ! 🍩
