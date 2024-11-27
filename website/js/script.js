const apiUrl = "http://127.0.0.1:5000/generate";
const chatContainer = document.getElementById("chat-container");
const inputElement = document.getElementById("user-input");
const sendButton = document.getElementById("send-button");

function typeMessage(content, type) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("message", type === "user" ? "user-message" : "ia-message");
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    let index = 0;
    function typeNextCharacter() {
        if (index < content.length) {
            messageDiv.textContent += content.charAt(index);
            index++;
            setTimeout(typeNextCharacter, 50);
        }
    }
    typeNextCharacter();
}

function sendMessage() {
    const userInput = inputElement.value.trim();
    if (!userInput) return;

    typeMessage(userInput, "user");

    fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: userInput, language: "en" }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) {
                typeMessage("Erreur : " + data.error, "ia");
            } else {
                typeMessage(data.generated, "ia");
            }
        })
        .catch(() => {
            typeMessage("Erreur de connexion.", "ia");
        });

    inputElement.value = "";
}

sendButton.addEventListener("click", sendMessage);
inputElement.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});
