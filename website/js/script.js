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
        addMessage(userMessage, "user");
        inputElement.value = "";
        // Simulating a bot response after a delay
        setTimeout(() => {
            addMessage("This is a bot response.", "bot");
        }, 1000);
    }
}

function addMessage(message, sender) {
    const messageElement = document.createElement("div");
    messageElement.classList.add("chat-message", sender);

    const messageContent = document.createElement("div");
    messageContent.classList.add("message-content");
    messageContent.textContent = message;

    messageElement.appendChild(messageContent);
    chatContainer.appendChild(messageElement);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}
