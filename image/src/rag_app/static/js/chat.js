document.addEventListener('DOMContentLoaded', () => {
    
    // 1. تعريف العناصر
    const chatBox = document.getElementById('chatBox');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const typingIndicator = document.getElementById('typingIndicator');

    // 2. إدارة الجلسة
    const sessionId = "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).substr(2, 9);
    console.log("Session ID:", sessionId);

    let chatHistory = []; 

    // ===========================================================
    // add new hook to open in new tab
    if (typeof DOMPurify !== 'undefined') {
        DOMPurify.addHook('afterSanitizeAttributes', function (node) {
            // التأكد أن العنصر هو رابط (Anchor Tag)
            if (node.tagName === 'A') {
                node.setAttribute('target', '_blank');
                node.setAttribute('rel', 'noopener noreferrer');
            }
        });
    }
    // ============================================================

    // دالة السكرول للأسفل
    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // دالة إنشاء عنصر الرسالة
    function createMessageElement(sender) {
        const div = document.createElement('div');
        div.className = `message ${sender}-message`;
        // الحيلة: إضافة الرسالة "قبل" المؤشر عشان المؤشر يفضل تحت
        chatBox.insertBefore(div, typingIndicator);
        return div;
    }

    // دالة عرض رسالة المستخدم
    function appendUserMessage(text) {
        const div = createMessageElement('user');
        div.textContent = text;
        scrollToBottom();
    }

    // الدالة الرئيسية للإرسال
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // تعطيل الإدخال
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;

        // 1. عرض رسالة المستخدم
        appendUserMessage(message);

        // 2. إظهار النقط فوراً (Loading)
        typingIndicator.style.display = 'block';
        scrollToBottom();

        try {
            // 3. الاتصال بالسيرفر
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory,
                    session_id: sessionId 
                })
            });

            if (!response.ok) throw new Error('Server error');

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let botMessageFull = '';
            
            // متغير للتحكم في ظهور الفقاعة (لن تظهر إلا مع أول حرف)
            let botMessageDiv = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                botMessageFull += chunk;

                // --- اللحظة الحاسمة ---
                // أول ما توصل داتا، نخفي النقط ونظهر الفقاعة
                if (!botMessageDiv) {
                    typingIndicator.style.display = 'none'; // اختفاء النقط
                    botMessageDiv = createMessageElement('bot'); // ظهور الفقاعة
                }

                // عرض النص (مع دعم Markdown والروابط الآمنة)
                botMessageDiv.innerHTML = DOMPurify.sanitize(marked.parse(botMessageFull));
                scrollToBottom(); 
            }

            // حفظ المحادثة في الذاكرة المؤقتة
            chatHistory.push([message, botMessageFull]);

        } catch (error) {
            console.error("Error:", error);
            typingIndicator.style.display = 'none';
            
            const errorDiv = createMessageElement('bot');
            errorDiv.textContent = "عذراً، حدث خطأ في الاتصال.";
            errorDiv.style.color = "red";
            
        } finally {
            // إعادة تفعيل الإدخال
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
            scrollToBottom();
        }
    }

    // تفعيل الأزرار
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // التركيز عند فتح الصفحة
    userInput.focus();
});