const local = "http://127.0.0.1:5000"; // URL de votre backend Flask
const inputElement = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");
const chatContainer = document.getElementById("chat-container");
const toggle = document.getElementById("toggle");

sendButton.addEventListener("click", sendMessage);
inputElement.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});

toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark-mode", toggle.checked);
});

function sendMessage() {
    const userMessage = inputElement.value.trim();
    if (userMessage !== "") {
        addMessage(userMessage, "user"); // Ajouter le message utilisateur
        inputElement.value = "";

        // Appel à l'API Flask
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
        .then(response => response.json())
        .then(data => {
            if (data.generated) {
                addMessage(data.generated, "bot"); // Ajouter le message bot
            } else if (data.error) {
                addMessage(`Error: ${data.error}`, "bot");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            addMessage("Sorry, something went wrong. Please try again.", "bot");
        });
    }
}

function addMessage(message, sender) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender);

    const messageContent = document.createElement("div");
    messageContent.classList.add("message-content");
    messageElement.appendChild(messageContent);
    chatContainer.appendChild(messageElement);

    // Faire défiler vers le bas automatiquement
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Ajouter une classe pour afficher l'effet de frappe
    messageContent.classList.add("typing");

    // Écrire le texte caractère par caractère
    let index = 0;
    const typingInterval = 50; // Intervalle en ms entre chaque caractère

    function typeCharacter() {
        if (index < message.length) {
            messageContent.textContent += message.charAt(index);
            index++;
            setTimeout(typeCharacter, typingInterval);
        } else {
            // Fin de l'animation : retirer l'effet de frappe
            messageContent.classList.remove("typing");
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    typeCharacter();
}

