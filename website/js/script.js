// URL de l'API Flask backend
const local = "http://127.0.0.1:5000"; 

// Éléments HTML interactifs
const inputElement = document.getElementById("user-input"); // Champ de saisie utilisateur
const sendButton = document.getElementById("send-button"); // Bouton d'envoi
const chatContainer = document.getElementById("chat-container"); // Conteneur des messages
const toggle = document.getElementById("toggle"); // Bouton pour mode sombre

// Événements : Envoi du message avec un clic ou la touche "Entrée"
sendButton.addEventListener("click", sendMessage);
inputElement.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});

// Changement du mode sombre
toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark-mode", toggle.checked);
});

// Fonction pour gérer l'envoi du message
function sendMessage() {
    const userMessage = inputElement.value.trim(); // Récupération et nettoyage du texte
    if (userMessage !== "") {
        addMessage(userMessage, "user"); // Ajout du message utilisateur
        inputElement.value = ""; // Réinitialisation du champ

        // Appel à l'API Flask pour générer une réponse
        fetch(local + '/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                input: userMessage,  // Texte utilisateur
                language: 'fr'       // Langue cible
            })
        })
        .then(response => response.json()) // Traitement de la réponse JSON
        .then(data => {
            if (data.generated) {
                addMessage(data.generated, "bot"); // Ajout du message bot généré
            } else if (data.error) {
                addMessage(`Error: ${data.error}`, "bot"); // Affichage des erreurs
            }
        })
        .catch(error => {
            console.error("Error:", error); // Gestion des erreurs réseau/API
            addMessage("Sorry, something went wrong. Please try again.", "bot");
        });
    }
}

// Fonction pour ajouter un message au chat
function addMessage(message, sender) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender); // Classe pour distinguer utilisateur/bot

    const messageContent = document.createElement("div");
    messageContent.classList.add("message-content");
    messageElement.appendChild(messageContent);
    chatContainer.appendChild(messageElement);

    // Faire défiler vers le bas pour voir les nouveaux messages
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Animation de frappe progressive
    messageContent.classList.add("typing");

    let index = 0; // Index actuel dans le texte
    const typingInterval = 50; // Vitesse de frappe (ms par caractère)

    function typeCharacter() {
        if (index < message.length) {
            messageContent.textContent += message.charAt(index); // Ajout du caractère
            index++;
            setTimeout(typeCharacter, typingInterval); // Prochain caractère
        } else {
            messageContent.classList.remove("typing"); // Fin de l'animation
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    typeCharacter();
}
