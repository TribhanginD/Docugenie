import React, { useRef, useEffect } from 'react';
import { Send, User, Sparkles, Quote, ExternalLink } from 'lucide-react';

const Message = ({ role, content, citations }) => {
    const isUser = role === 'user';

    return (
        <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'} animate-slide-up`}>
            {!isUser && (
                <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center shrink-0 shadow-lg shadow-brand-600/20">
                    <Sparkles size={16} className="text-white" />
                </div>
            )}
            <div className={`max-w-[80%] space-y-3 ${isUser ? 'order-1' : 'order-2'}`}>
                <div className={`p-4 rounded-2xl ${isUser
                    ? 'bg-brand-600 text-white rounded-tr-none shadow-xl shadow-brand-600/10'
                    : 'glass-panel rounded-tl-none'
                    }`}>
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
                </div>

                {citations && citations.length > 0 && (
                    <div className="flex flex-wrap gap-2 pl-1">
                        {citations.map((cite, i) => (
                            <div
                                key={i}
                                className="flex items-center gap-1.5 px-2 py-1 rounded bg-white/5 border border-white/5 text-[10px] text-white/50 hover:text-white/80 hover:bg-white/10 transition-all cursor-pointer group"
                            >
                                <Quote size={10} className="text-brand-400" />
                                <span>{cite.doc} (p. {cite.page || '?'})</span>
                                <ExternalLink size={8} className="opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                        ))}
                    </div>
                )}
            </div>
            {isUser && (
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center shrink-0 border border-white/10 order-2">
                    <User size={16} className="text-white/60" />
                </div>
            )}
        </div>
    );
};

const Chat = ({ messages, input, setInput, handleSend, isLoading }) => {
    const scrollRef = useRef(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isLoading]);

    return (
        <div className="flex-1 flex flex-col h-screen bg-transparent relative overflow-hidden">
            {/* Background Decor */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-brand-600/5 blur-[120px] rounded-full pointer-events-none" />

            {/* Messages */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-8 space-y-8 scroll-smooth"
            >
                {messages.length === 0 && (
                    <div className="h-full flex flex-col items-center justify-center space-y-4 animate-fade-in">
                        <div className="w-16 h-16 bg-white/5 rounded-2xl flex items-center justify-center border border-white/10">
                            <Sparkles className="text-brand-400" size={32} />
                        </div>
                        <div className="text-center space-y-1">
                            <h2 className="text-xl font-bold">How can I help you today?</h2>
                            <p className="text-sm text-white/40">Upload your PDFs and ask anything about them.</p>
                        </div>
                    </div>
                )}
                {messages.map((msg, i) => (
                    <Message key={i} {...msg} />
                ))}
                {isLoading && (
                    <div className="flex gap-4 animate-pulse-soft">
                        <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center shrink-0">
                            <Sparkles size={16} className="text-white" />
                        </div>
                        <div className="glass-panel p-4 rounded-2xl rounded-tl-none w-12 flex justify-center">
                            <span className="flex gap-1">
                                <span className="w-1 h-1 bg-white/40 rounded-full animate-bounce" />
                                <span className="w-1 h-1 bg-white/40 rounded-full animate-bounce [animation-delay:0.2s]" />
                                <span className="w-1 h-1 bg-white/40 rounded-full animate-bounce [animation-delay:0.4s]" />
                            </span>
                        </div>
                    </div>
                )}
            </div>

            {/* Input */}
            <div className="p-8 pt-0">
                <form
                    onSubmit={handleSend}
                    className="relative max-w-4xl mx-auto group animate-slide-up"
                >
                    <input
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        disabled={isLoading}
                        placeholder="Ask DocuGenie something..."
                        className="w-full bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl px-6 py-4 pr-16 text-sm focus:outline-none focus:border-brand-500/50 transition-all shadow-2xl group-hover:bg-white/10"
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="absolute right-3 top-2.5 w-10 h-10 bg-brand-600 hover:bg-brand-500 disabled:bg-white/5 disabled:text-white/20 rounded-xl flex items-center justify-center transition-all text-white shadow-lg shadow-brand-600/20 active:scale-95"
                    >
                        <Send size={18} />
                    </button>
                </form>
                <p className="text-center text-[10px] text-white/20 mt-4 uppercase tracking-widest font-bold">
                    AI generated content can be incorrect. Verify citations.
                </p>
            </div>
        </div>
    );
};

export default Chat;
