import React, { useState } from 'react';
import { Settings, Key, Cpu, Trash2, Github } from 'lucide-react';

const APIKeyInput = ({ choice, apiKey, setApiKey }) => {
    const [show, setShow] = useState(false);

    return (
        <div className="space-y-2 animate-slide-up">
            <div className="flex items-center gap-2 text-sm text-white/60 mb-1">
                <Key size={14} />
                <span>{choice} API Key (Optional)</span>
            </div>
            <div className="relative group">
                <input
                    type={show ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder={`Enter your ${choice} key...`}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-brand-500/50 transition-colors"
                />
                <button
                    onClick={() => setShow(!show)}
                    className="absolute right-2 top-1.5 text-white/30 hover:text-white/60 transition-colors"
                >
                    {show ? <span className="text-[10px] font-bold">HIDE</span> : <span className="text-[10px] font-bold">SHOW</span>}
                </button>
            </div>
            <p className="text-[10px] text-white/40 italic">
                If blank, server defaults will be used.
            </p>
        </div>
    );
};

export default APIKeyInput;
