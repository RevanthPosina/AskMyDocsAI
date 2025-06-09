css = '''
<style>
body {
    background: #181a20;
}
.chat-message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1.2rem;
    max-width: 90vw;
    word-break: break-word;
}
.chat-message.user {
    flex-direction: row-reverse;
    justify-content: flex-end;
}
.chat-message.bot {
    justify-content: flex-start;
}
.chat-message .avatar {
    flex-shrink: 0;
    width: 44px;
    height: 44px;
    margin: 0 0.7rem;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-message .avatar img {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    object-fit: cover;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    border: 2px solid #23272f;
}
.chat-message .message {
    padding: 1.1rem 1.3rem;
    border-radius: 1.2rem;
    font-size: 1.08rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    max-width: 600px;
    min-width: 60px;
    background: linear-gradient(90deg, #23272f 0%, #23272f 100%);
    color: #fff;
    margin-top: 0.1rem;
}
.chat-message.user .message {
    background: linear-gradient(90deg, #6a85b6 0%, #bac8e0 100%);
    color: #fff;
    text-align: right;
}
.chat-message.bot .message {
    background: linear-gradient(90deg, #23272f 0%, #23272f 100%);
    color: #fff;
    text-align: left;
}
@media (max-width: 700px) {
    .chat-message .message {
        font-size: 0.98rem;
        padding: 0.8rem 1rem;
        max-width: 90vw;
    }
    .chat-message .avatar {
        width: 36px;
        height: 36px;
    }
    .chat-message .avatar img {
        width: 36px;
        height: 36px;
    }
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/VcBMzSW9/Hinabot.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/4wkQjfnV/user.png" alt="User Avatar">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
