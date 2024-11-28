# GPT-NSI

![License](https://img.shields.io/github/license/Mizmalik/GPT-NSI)  
![Issues](https://img.shields.io/github/issues/Mizmalik/GPT-NSI)  
![Stars](https://img.shields.io/github/stars/Mizmalik/GPT-NSI)  

GPT-NSI est un projet conçu pour exploiter les capacités avancées du traitement du langage naturel (NLP) dans le cadre des **Numériques et Sciences Informatiques (NSI)**. Il s'adresse particulièrement aux étudiants, enseignants et développeurs travaillant sur des sujets éducatifs ou pratiques en lien avec l'informatique.

Ce dépôt contient des ressources, du code et des outils basés sur des systèmes d'IA comme GPT, pour aider à mieux comprendre et résoudre des problématiques liées au programme NSI.

## Fonctionnalités

- **Assistance basée sur l'IA** : Exploitez des modèles basés sur GPT pour expliquer, résoudre ou générer du code et des concepts liés au programme NSI.
- **Base de code personnalisable** : Adaptez les modèles GPT et leur comportement à vos besoins spécifiques.
- **Focus éducatif** : Contenu spécialement conçu pour aider dans les domaines du programme NSI tels que la programmation, les algorithmes et la résolution de problèmes.
- **Open Source** : Contributions bienvenues pour enrichir le projet.

## Prise en main

Suivez les étapes ci-dessous pour installer et utiliser GPT-NSI sur votre machine locale :

### Prérequis

- Python 3.8 ou une version ultérieure
- `pip` (gestionnaire de paquets Python)
- Une clé API GPT (par exemple, OpenAI ou autre système compatible)

### Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/Mizmalik/GPT-NSI.git
   cd GPT-NSI
   ```

2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

3. Configurez votre clé API :
   - Créez un fichier `.env` à la racine du projet.
   - Ajoutez la ligne suivante dans le fichier :
     ```
     API_KEY=your_openai_api_key
     ```

4. Lancez l’application :
   ```bash
   python main.py
   ```

### Utilisation

- L'application propose une interface en ligne de commande (CLI) et/ou une interface web pour interagir avec le modèle GPT.
- Posez des questions ou saisissez des problèmes liés au programme NSI pour obtenir des solutions générées par l'IA, des extraits de code ou des explications détaillées.

## Contribution

Les contributions sont les bienvenues ! Suivez ces étapes pour contribuer :

1. **Forkez** le dépôt.
2. Créez une nouvelle branche :
   ```bash
   git checkout -b feature/votre-nouvelle-fonctionnalité
   ```
3. Apportez vos modifications et validez-les :
   ```bash
   git commit -m "Ajout de votre fonctionnalité"
   ```
4. Poussez vos modifications :
   ```bash
   git push origin feature/votre-nouvelle-fonctionnalité
   ```
5. Soumettez une **pull request**.

## Licence

Ce projet est sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour plus de détails.
