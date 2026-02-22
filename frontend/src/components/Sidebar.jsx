import React from 'react';
import { FileUp, FileText, Trash2, Loader2, Database, Wand2, ExternalLink } from 'lucide-react';
import APIKeyInput from './APIKeyInput';

const Sidebar = ({
    provider,
    setProvider,
    apiKey,
    setApiKey,
    files,
    handleUpload,
    isProcessing,
    useContextual,
    setUseContextual,
    useHyDE,
    setUseHyDE
}) => {
    return (
        <aside className="w-80 h-screen glass-panel border-r border-white/10 flex flex-col p-6 overflow-y-auto z-10">
            <div className="flex items-center gap-3 mb-8">
                <div className="w-10 h-10 bg-brand-600 rounded-xl flex items-center justify-center shadow-lg shadow-brand-600/20">
                    <Wand2 className="text-white" size={24} />
                </div>
                <div>
                    <h1 className="font-bold text-xl tracking-tight">DocuGenie</h1>
                    <p className="text-[10px] text-brand-400 font-bold uppercase tracking-wider">v2.0 Production</p>
                </div>
            </div>

            <div className="space-y-8 flex-1">
                {/* Provider Selection */}
                <section className="space-y-4">
                    <div className="flex items-center gap-2 text-xs font-bold text-white/40 uppercase tracking-widest">
                        <Database size={14} />
                        <span>Model Provider</span>
                    </div>
                    <select
                        value={provider}
                        onChange={(e) => setProvider(e.target.value)}
                        className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-brand-500/50 transition-colors appearance-none cursor-pointer"
                    >
                        <option value="Groq" className="bg-[#1a1a1e]">Groq (Fastest)</option>
                        <option value="Google Gemini" className="bg-[#1a1a1e]">Google Gemini (Advanced)</option>
                    </select>

                    <APIKeyInput choice={provider} apiKey={apiKey} setApiKey={setApiKey} />
                </section>

                {/* File Upload */}
                <section className="space-y-4">
                    <div className="flex items-center gap-2 text-xs font-bold text-white/40 uppercase tracking-widest">
                        <FileUp size={14} />
                        <span>Knowledge Base</span>
                    </div>
                    <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-white/10 rounded-xl cursor-pointer hover:border-brand-500/30 hover:bg-white/5 transition-all group">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6">
                            <FileUp className="text-white/30 group-hover:text-brand-400 transition-colors mb-2" size={24} />
                            <p className="text-xs text-white/40 group-hover:text-white/60">Upload PDFs</p>
                        </div>
                        <input type="file" className="hidden" multiple accept=".pdf" onChange={handleUpload} />
                    </label>

                    {/* File List */}
                    <div className="space-y-2 max-h-48 overflow-y-auto pr-2">
                        {files.map((file, i) => (
                            <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-white/5 border border-white/5 group animate-fade-in">
                                <div className="flex items-center gap-2 truncate">
                                    <FileText size={14} className="text-brand-400 shrink-0" />
                                    <span className="text-xs truncate">{file.name}</span>
                                </div>
                                {isProcessing ? (
                                    <Loader2 size={12} className="animate-spin text-white/20" />
                                ) : (
                                    <button className="text-white/20 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100">
                                        <Trash2 size={12} />
                                    </button>
                                )}
                            </div>
                        ))}
                    </div>
                </section>

                {/* Advanced Toggles */}
                <section className="space-y-4 pt-4 border-t border-white/5">
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-white/60">Contextual Retrieval</span>
                        <button
                            onClick={() => setUseContextual(!useContextual)}
                            className={`w-8 h-4 rounded-full transition-colors relative ${useContextual ? 'bg-brand-500' : 'bg-white/10'}`}
                        >
                            <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${useContextual ? 'translate-x-4' : 'translate-x-0'}`} />
                        </button>
                    </div>
                    <div className="flex items-center justify-between">
                        <span className="text-xs text-white/60">HyDE</span>
                        <button
                            onClick={() => setUseHyDE(!useHyDE)}
                            className={`w-8 h-4 rounded-full transition-colors relative ${useHyDE ? 'bg-brand-500' : 'bg-white/10'}`}
                        >
                            <div className={`absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform ${useHyDE ? 'translate-x-4' : 'translate-x-0'}`} />
                        </button>
                    </div>
                </section>
            </div>

            <div className="mt-auto pt-6 border-t border-white/5">
                <a
                    href="https://github.com"
                    target="_blank"
                    className="flex items-center gap-2 text-[10px] text-white/30 hover:text-white/60 transition-colors"
                >
                    <ExternalLink size={12} />
                    <span>Documentation & Support</span>
                </a>
            </div>
        </aside>
    );
};

export default Sidebar;
