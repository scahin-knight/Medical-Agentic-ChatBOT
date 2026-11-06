const $ = id => document.getElementById(id);

let imageData = null;
let cameraStream = null;
let lastAnswer = '';

// Local CPU answers can be slow, so allow up to 5 minutes
const REQUEST_TIMEOUT_MS = 300000;

// --------------------------------------------------
// Conversation ID (valid UUID, kept until "New chat")
// --------------------------------------------------

function makeId() {
    if (crypto.randomUUID) return crypto.randomUUID();

    // Fallback for non-HTTPS pages
    const b = crypto.getRandomValues(new Uint8Array(16));
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const h = [...b].map(x => x.toString(16).padStart(2, '0')).join('');
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

let conversationId = localStorage.getItem('medguide_conversation_id');

if (!conversationId) {
    conversationId = makeId();
    localStorage.setItem('medguide_conversation_id', conversationId);
}

// Keep the token only for this browser tab session
try {
    $('token').value = sessionStorage.getItem('medguide_token') || '';
} catch {}

$('token').addEventListener('change', () => {
    try {
        sessionStorage.setItem('medguide_token', $('token').value.trim());
    } catch {}
});

const headers = () => {
    const h = { 'X-Conversation-ID': conversationId };
    const token = $('token').value.trim();
    if (token) h.Authorization = `Bearer ${token}`;
    return h;
};

// --------------------------------------------------
// Helpers
// --------------------------------------------------

async function fetchWithTimeout(url, options = {}, ms = REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ms);

    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

async function readError(r) {
    try {
        const d = await r.json();
        if (typeof d.detail === 'string') return d.detail;
        if (Array.isArray(d.detail)) return d.detail.map(x => x.msg).join('; ');
    } catch {}
    return `Request failed (${r.status})`;
}

function friendlyError(err) {
    if (err.name === 'AbortError') {
        return 'The request took too long. Please try again.';
    }
    return err.message;
}

function renderSources(el, sources) {
    if (!sources || !sources.length) return;

    const box = document.createElement('div');
    box.className = 'sources';
    box.append('Sources: ');

    const seen = new Set();

    for (const s of sources) {
        const key = s.type === 'pdf' ? `pdf-${s.page}` : s.url;
        if (!key || seen.has(key)) continue;
        if (seen.size >= 8) break;

        if (seen.size > 0) box.append(' | ');
        seen.add(key);

        if (s.type === 'pdf') {
            box.append(`PDF page ${s.page}`);
        } else if (/^https?:\/\//i.test(s.url || '')) {
            const a = document.createElement('a');
            a.href = s.url;
            a.textContent = s.title || s.url;
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            box.append(a);
        } else {
            box.append(s.title || 'Web source');
        }
    }

    if (seen.size) el.appendChild(box);
}

function bubble(text, kind = 'bot') {
    const el = document.createElement('article');
    el.className = kind;
    el.textContent = text;

    $('messages').appendChild(el);
    $('messages').scrollTop = $('messages').scrollHeight;

    return el;
}

// --------------------------------------------------
// Health check
// --------------------------------------------------

async function check() {
    try {
        const r = await fetch('/api/health');
        const d = await r.json();

        if (!d.ollama_running) {
            $('status').textContent = '⚠ Start Ollama';
        } else if (!d.pinecone_configured) {
            $('status').textContent = '⚠ Configure API keys';
        } else {
            $('status').textContent = '● Connected';
        }
    } catch {
        $('status').textContent = '● Server offline';
    }
}

check();

// --------------------------------------------------
// Image selection
// --------------------------------------------------

function setImage(file) {
    if (!file) return;

    if (
        !['image/png', 'image/jpeg', 'image/webp'].includes(file.type) ||
        file.size > 5 * 1024 * 1024
    ) {
        alert('Use PNG/JPEG/WebP under 5MB');
        return;
    }

    const reader = new FileReader();

    reader.onload = () => {
        imageData = reader.result;
        $('thumbnail').src = imageData;
        $('preview').hidden = false;
    };

    reader.readAsDataURL(file);
}

$('image').addEventListener('change', e => setImage(e.target.files[0]));

$('remove').onclick = () => {
    imageData = null;
    $('preview').hidden = true;
    $('thumbnail').src = '';
    $('image').value = '';
};

// --------------------------------------------------
// Chat
// --------------------------------------------------

$('chatForm').addEventListener('submit', async e => {
    e.preventDefault();

    const message = $('message').value.trim();
    if (!message && !imageData) return;

    const submittedImage = imageData;

    bubble(message || 'Please analyse this image.', 'user');

    $('message').value = '';
    $('send').disabled = true;

    const pending = bubble('Thinking… (local model, this can take a minute)', 'bot thinking');

    try {
        const r = await fetchWithTimeout('/api/chat', {
            method: 'POST',
            headers: {
                ...headers(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message || 'Please analyse this medical image.',
                image_data: submittedImage
            })
        });

        if (!r.ok) throw new Error(await readError(r));

        const d = await r.json();

        pending.classList.remove('thinking');
        pending.textContent = d.answer || 'No answer received.';
        lastAnswer = d.answer || '';

        renderSources(pending, d.sources);

        if (submittedImage === imageData) {
            $('remove').click();
        }

    } catch (err) {
        pending.classList.remove('thinking');
        pending.classList.add('error');
        pending.textContent = 'Error: ' + friendlyError(err);
    } finally {
        $('send').disabled = false;
        $('messages').scrollTop = $('messages').scrollHeight;
    }
});

// Enter sends, Shift+Enter makes a new line
$('message').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        $('chatForm').requestSubmit();
    }
});

// --------------------------------------------------
// New chat
// --------------------------------------------------

$('newChat').onclick = () => {
    if ('speechSynthesis' in window) speechSynthesis.cancel();

    conversationId = makeId();
    localStorage.setItem('medguide_conversation_id', conversationId);

    $('messages').replaceChildren();
    lastAnswer = '';

    bubble('New conversation started. How can I help?');
};

// --------------------------------------------------
// Voice input (browser speech recognition, Chrome/Edge)
// --------------------------------------------------

const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

$('mic').onclick = () => {
    if (!SpeechRec) {
        alert('Voice input needs Chrome or Edge.');
        return;
    }

    if (recognition) {
        recognition.stop();
        return;
    }

    recognition = new SpeechRec();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;

    recognition.onresult = e => {
        const text = [...e.results].map(r => r[0].transcript).join(' ');
        $('message').value = ($('message').value + ' ' + text).trim();
    };

    recognition.onerror = e => {
        if (e.error !== 'aborted' && e.error !== 'no-speech') {
            alert('Voice input error: ' + e.error);
        }
    };

    recognition.onend = () => {
        recognition = null;
        $('mic').textContent = '🎤 Voice';
    };

    recognition.start();
    $('mic').textContent = '⏹ Stop recording';
};

// --------------------------------------------------
// Read answer aloud (browser speech synthesis)
// --------------------------------------------------

$('listen').onclick = () => {
    if (!('speechSynthesis' in window)) {
        alert('Read-aloud is not supported in this browser.');
        return;
    }

    if (speechSynthesis.speaking) {
        speechSynthesis.cancel();
        return;
    }

    if (!lastAnswer) {
        alert('Ask a question first');
        return;
    }

    // Remove citations and markdown symbols so they aren't read aloud
    const speakable = lastAnswer
        .replace(/\[PDF p\.\d+\]/g, '')
        .replace(/https?:\/\/\S+/g, '')
        .replace(/[*_#`>]/g, '')
        .replace(/\s+/g, ' ')
        .trim();

    if (!speakable) return;

    const utter = new SpeechSynthesisUtterance(speakable);
    utter.lang = 'en-IN';
    speechSynthesis.speak(utter);
};

// --------------------------------------------------
// Camera
// --------------------------------------------------

$('camera').onclick = async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });

        $('video').srcObject = cameraStream;
        $('video').hidden = false;
        $('cameraActions').hidden = false;

    } catch (err) {
        alert('Camera unavailable: ' + err.message);
    }
};

$('closeCamera').onclick = () => {
    cameraStream?.getTracks().forEach(t => t.stop());
    cameraStream = null;

    $('video').srcObject = null;
    $('video').hidden = true;
    $('cameraActions').hidden = true;
};

$('snap').onclick = () => {
    const v = $('video');
    const c = $('canvas');

    if (!v.videoWidth || !v.videoHeight) {
        alert('Camera is not ready yet.');
        return;
    }

    // Downscale so photos stay under the 5 MB limit
    const maxSide = 1280;
    const scale = Math.min(1, maxSide / Math.max(v.videoWidth, v.videoHeight));

    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);

    c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);

    c.toBlob(blob => {
        if (blob) {
            setImage(new File([blob], 'camera.jpg', { type: 'image/jpeg' }));
        }
    }, 'image/jpeg', 0.85);

    $('closeCamera').click();
};

// --------------------------------------------------
// Cleanup
// --------------------------------------------------

window.addEventListener('beforeunload', () => {
    cameraStream?.getTracks().forEach(t => t.stop());
    if ('speechSynthesis' in window) speechSynthesis.cancel();
});