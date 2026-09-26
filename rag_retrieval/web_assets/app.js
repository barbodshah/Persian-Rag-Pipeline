const state = {
  imageData: null,
  imageName: null,
  history: [],
  details: [],
  selectedDetail: null,
  activeTab: "context",
  sending: false,
};

const shell = document.getElementById("app-shell");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const composer = document.getElementById("composer");
const input = document.getElementById("message-input");
const imageInput = document.getElementById("image-input");
const preview = document.getElementById("attachment-preview");
const previewImage = document.getElementById("preview-image");
const previewName = document.getElementById("preview-name");
const previewSize = document.getElementById("preview-size");
const sendButton = document.getElementById("send-button");
const inspector = document.getElementById("inspector");
const inspectorContent = document.getElementById("inspector-content");
const inspectorSubtitle = document.getElementById("inspector-subtitle");
const toggleInspector = document.getElementById("toggle-inspector");
const bookSelect = document.getElementById("book-select");
const modeSelect = document.getElementById("mode-select");

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function setInspector(open) {
  inspector.hidden = !open;
  shell.classList.toggle("inspector-open", open);
  toggleInspector.setAttribute("aria-expanded", String(open));
  toggleInspector.textContent = open ? "Hide details" : "Show details";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clearImage() {
  state.imageData = null;
  state.imageName = null;
  imageInput.value = "";
  preview.hidden = true;
  previewImage.removeAttribute("src");
}

function addMessage(role, text, options = {}) {
  welcome.hidden = true;
  const row = document.createElement("article");
  row.className = `message ${role}${options.error ? " error" : ""}${options.thinking ? " thinking" : ""}`;
  const card = document.createElement("div");
  card.className = "message-card";
  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (options.image) {
    const image = document.createElement("img");
    image.className = "message-image";
    image.src = options.image;
    image.alt = "Question screenshot";
    bubble.appendChild(image);
  }
  const content = document.createElement("span");
  content.textContent = text;
  if (options.thinking) content.className = "dots";
  bubble.appendChild(content);
  card.appendChild(bubble);

  if (Number.isInteger(options.detailIndex)) {
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "detail-link";
    button.textContent = "Inspect pipeline details";
    button.addEventListener("click", () => {
      state.selectedDetail = options.detailIndex;
      renderInspector();
      setInspector(true);
    });
    meta.appendChild(button);
    card.appendChild(meta);
  }

  row.appendChild(card);
  messages.appendChild(row);
  scrollToBottom();
  return row;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderContext(detail) {
  if (!detail.retrievals?.length) {
    inspectorContent.appendChild(element("div", "empty-detail", "No retrieval context returned."));
    return;
  }
  for (const retrieval of detail.retrievals) {
    const block = element("section", "detail-block");
    block.appendChild(element("h3", "", `${retrieval.query_id} · ${retrieval.query_text}`));
    if (!retrieval.passages.length) {
      block.appendChild(element("p", "", "No passage was retrieved for this query."));
    }
    for (const passage of retrieval.passages) {
      const card = element("article", "passage");
      card.appendChild(element("strong", "", passage.citation_id));
      const pages = passage.book_pages?.join(", ") || "—";
      card.appendChild(element("small", "", `${passage.chapter_title} · pages ${pages}`));
      card.appendChild(element("div", "", passage.text));
      block.appendChild(card);
    }
    inspectorContent.appendChild(block);
  }
}

function renderInspector() {
  inspectorContent.replaceChildren();
  const detail = state.details[state.selectedDetail];
  if (!detail) {
    inspectorContent.appendChild(element("div", "empty-detail", "Send a question to inspect its pipeline output."));
    return;
  }
  inspectorSubtitle.textContent = `Answer ${state.selectedDetail + 1}`;
  if (state.activeTab === "context") {
    renderContext(detail);
  } else if (state.activeTab === "ocr") {
    const ocr = detail.input?.ocr_text || "No screenshot was used for this message.";
    inspectorContent.appendChild(element("pre", "code-output", ocr));
  } else {
    inspectorContent.appendChild(element("pre", "code-output json", JSON.stringify(detail.structure, null, 2)));
  }
}

async function loadBooks() {
  try {
    const response = await fetch("/api/books");
    const data = await response.json();
    for (const book of data.books || []) {
      const option = document.createElement("option");
      option.value = book;
      option.textContent = book;
      bookSelect.appendChild(option);
    }
  } catch (_) {
    addMessage("assistant", "Could not load the available books.", { error: true });
  }
}

async function submitQuestion() {
  const message = input.value.trim();
  if (state.sending || (!message && !state.imageData)) return;
  state.sending = true;
  sendButton.disabled = true;

  const sentImage = state.imageData;
  const sentImageName = state.imageName;
  addMessage("user", message || "Question in the attached screenshot", { image: sentImage });
  input.value = "";
  input.style.height = "auto";
  clearImage();
  const thinking = addMessage("assistant", "•••", { thinking: true });

  try {
    const response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        image_data: sentImage,
        image_name: sentImageName,
        book_id: bookSelect.value || null,
        mode: modeSelect.value,
        history: state.history,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed with status ${response.status}`);
    thinking.remove();
    const detailIndex = state.details.push(data) - 1;
    state.selectedDetail = detailIndex;
    addMessage("assistant", data.answer, { detailIndex });
    const resolvedUserText = [data.input?.typed_text, data.input?.ocr_text].filter(Boolean).join("\n\n");
    state.history.push({ role: "user", content: resolvedUserText || message });
    state.history.push({ role: "assistant", content: data.answer });
    state.history = state.history.slice(-8);
    renderInspector();
  } catch (error) {
    thinking.remove();
    addMessage("assistant", error.message || "The request failed.", { error: true });
  } finally {
    state.sending = false;
    sendButton.disabled = false;
    input.focus();
  }
}

imageInput.addEventListener("change", () => {
  const file = imageInput.files?.[0];
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) {
    addMessage("assistant", "Please choose an image smaller than 10 MB.", { error: true });
    clearImage();
    return;
  }
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    state.imageData = reader.result;
    state.imageName = file.name;
    previewImage.src = reader.result;
    previewName.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    preview.hidden = false;
  });
  reader.readAsDataURL(file);
});

composer.addEventListener("submit", event => {
  event.preventDefault();
  submitQuestion();
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

document.getElementById("remove-image").addEventListener("click", clearImage);
toggleInspector.addEventListener("click", () => setInspector(inspector.hidden));
document.getElementById("close-inspector").addEventListener("click", () => setInspector(false));

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    state.activeTab = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach(item => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    renderInspector();
  });
});

document.getElementById("new-chat").addEventListener("click", () => {
  state.history = [];
  state.details = [];
  state.selectedDetail = null;
  clearImage();
  messages.querySelectorAll(".message").forEach(item => item.remove());
  welcome.hidden = false;
  inspectorSubtitle.textContent = "Latest answer";
  renderInspector();
  input.focus();
});

loadBooks();
