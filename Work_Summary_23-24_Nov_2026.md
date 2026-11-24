# Kaam jo hua — 23 se 24 Nov 2026

Ishaq — kal se ab tak jo sab hua, ye uski poori list hai.

## 23 Nov 2026

### PhD — fully funded RA / TA
- Goal set kiya: fully funded PhD scholarship RA ya TA jeetna hai.
- Rule: USA professor ko direct email nahi karna, pehle unka kaam dekhna hai, limitations check karni hain, phir future work.
- Prof. Cheng Zhang (Texas A&M) ko 13 Nov ko email bheja tha TaxaDiffusion paper par. Zhang ne 23 Nov subah 7:48 AM reply kiya: Fall 2027 ke liye PhD students hire kar rahe hain, GenAI aur 3DV par focus hai, application mein apna naam SOP aur system dono mein likhne ko kaha, interviews Dec end / Jan start mein.
- Zhang pilot complete hua: repo, LinkedIn post, aur thank-you email 23 Nov ko bheja gaya.
- Prof. Dongyeop Kang (UMN CS) ne 23 Nov 20:02 UTC ko YES reply kiya: Fall 2027 ke liye kuch openings hain, UMN CS PhD apply karne aur unko potential advisor list karne ko kaha. Deadlines: 15 Dec 2026 fellowship consideration, 5 Jan 2027 final. Fee $95. Total TAMU ($148) + UMN ($95) = $243. Thank-you draft + CLIP pilot plan tayyar hai, email abhi nahi bheja — tumhari approval chahiye.
- Texas A&M deadline 15 Dec 2026, 3 recommendation letters chahiye, koi recommender abhi naam nahi hua.

### Event-SAE — shuruaat
- Notebook: Event-SAE-Colab-Setup.ipynb, 56 cells, Drive file ID 1ZXUnLytb7qKneTjtI5fGct8PIUA6iF7m.
- Goal: NeurIPS 2026 paper 'Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies' reproduce karna — OpenVLA 7B + LIBERO-Goal, paper baseline 68% → top-5 event-aligned features zero karne par 48.8%.
- Free Colab check kiya: L4 nahi mila, T4 mila. Cell 21 re-run 21:52 pe shuru hua, runtime 22:14 pe disconnect ho gaya. Episode 0, 1, 2 success, episode 3 fail, episode 4 load hote hue disconnect.
- Us raat kaam roka gaya, Kaggle try karne ko kaha gaya — Kaggle sign-in nahi tha, is liye GPU check ruk gaya.

## 24 Nov 2026

### Event-SAE — L4 run, poora complete
- Subah 06:11 pe L4-only try kiya, 06:15 pe L4 connect hua: NVIDIA L4, 23,034 MiB, torch 2.14.1+cu130.
- Setup dobara banaya: clone, installs, downloads (SAE ae.pt 1.07 GB + OpenVLA 16 files ~14 GB), LIBERO patch, osmesa se EGL switch, render test OK (64, 64, 3).
- Cell 21 rollout: 06:34 se 08:48 tak, 50/50 complete, **35/50 = 70.0%**, 8,020.341s.
- extract_topk (chunked Cell 55): 09:26 pe complete, **2,426,654 rows**, manifest likha gaya.
- Cell 25 keyframes: 187 samples, 09:48 pe.
- Cell 26 event features + clustering: 187 events, 10 tasks, **42 clusters**, 09:50 pe.
- Cell 28 scoring: rank-1 event-aligned feature **16646**, 09:58 pe.
- Cell 30 steering configs: colab_native, colab_event, colab_random, 10:03 pe.
- Cell 32 baseline batch: **14/20 = 70.0%**, 10:56 pe, 3,088.556s.
- Cell 33 event-aligned batch: **2/20 = 10.0%**, 12:26 pe, 2,363.942s, feature 16646 zero karke.
- Cell 34 random batch: **14/20 = 70.0%**, 13:27 pe, 3,402.047s.
- Cell 36 results table: 13:33 pe — **Baseline 70.0%, Event-aligned 10.0%, Random 70.0%** vs paper 68.0% / 48.8% / 68.7%. Direction check pass hua, event effect paper se zyada strong hai.
- Cell 39 zip: 13:33 se chala, **19:11 pe finish hua, 36G, 998.734s**, final line 'Download event-sae-results.zip from the file browser on the left.'
- Cell 57 Drive save: pehli try 19:42 pe fail hui 'ValueError: mount failed'. Dobara 19:51–19:54 pe chala, **'SIZE MATCH - saved to Drive OK'**, 37,693,297,786 bytes, MyDrive/event-sae-results.zip, Drive file ID 1ns6zEnEHnOTeERHA1H0dNDbtaSaKHuRY.
- Drive space check: total 5TB, 54.7GB use hua hai, ~4.95TB free hai.
- Professor kaun hai, ye abhi tak confirm nahi hua. Gmail mein do candidates mile:
  - Prof. Sergey Levine, UC Berkeley — OpenVLA paper, 10 Nov ko email bheja tha, koi reply nahi mila.
  - Prof. Dimitris Metaxas, Rutgers — LUCID-SAE paper, 11 Nov ko email bheja tha, koi reply nahi mila.

### AI VTuber — Mao
- 100% free Open-LLM-VTuber setup complete hua: Ollama 0.40.17, llama3.1:8b (4.9 GB), Python 3.12.10, Git, uv, ffmpeg, clone C:\Open-LLM-VTuber, uv sync OK.
- Server running hai http://localhost:12393 par, Mao avatar dikh raha hai, text replies kaam karte hain.
- Voice nahi kaam karta: Edge TTS 403 'blocked in your region'. Piper TTS switch ka plan tayyar hai, decision pending hai.
- OBS setup complete: Game Capture GTA V, Mao top-right mein, preview mein GTA V map dikh raha hai.
- Pehli YouTube live plan hai Wed 25 Nov ~11:25 PKT, channel Jokes Puncher, Stream Key, Public, title abhi decide nahi hua.

### Dartmouth Guarini info session — 7:00 PM
- Hector Sanchez (Assistant Dean) ne present kiya, Jane Seibel chat dekh rahi thi.
- 19 PhD programs, stipend $49,862 + health/dental + $1,500 relocation, 5 saal guaranteed pending lab match.
- Rotation-based: CSMP, Health Policy, Integrative Neuroscience, Quantitative Biomedical Sciences, MCB. Baaki direct admit hain, Computer Science direct admit hai.
- Personal statement 1,000 words max. GRE sirf 4 programs mein chahiye: Cognitive Neuroscience, Earth and Planetary Sciences, Health Policy, Psychological and Brain Sciences.
- Deadlines zyada tar 1 Dec aur 15 Dec 2026.
- Application fee waiver QR code / link end mein diya gaya, form attendance se linked hai, same email se bharna hai jis se register kiya tha. Form bhara ya nahi, ye confirm nahi hai.
- Poori chat 7:00–7:43 PM save ki gayi hai.

### GTA V
- Official purchase breakdown diya gaya: Steam $29.99, Epic $29.99, Rockstar Store Rs 6,900. Sale $14.99, last sale 5 Nov ko khatam hui, next Winter Sale 17 Dec–4 Jan.
- Gamesplanet $13.99 sab se sasta official option hai, tum us taraf lean kar rahe ho, purchase abhi confirm nahi hua.

### Facebook
- Page removal appeal tayyar hai (8 Nov ko remove hua tha 'Inauthentic Behavior' par), corrected version file mein hai.
- Payout account restricted hai, 'Request a review' button form nahi kholta, review abhi submit nahi hua.

### GitHub repo + LinkedIn — 24 Nov shaam
- README.md, RESULTS.md, results-chart.png, aur LinkedIn post draft tayyar hain, local folder ~/workspace/phd-project/event-sae-demo/ mein.
- Chart: Baseline 70.0%, Event-aligned 10.0%, Random 70.0% vs paper 68.0% / 48.8% / 68.7%.
- Equations README mein hain: SAE reconstruction, loss, TopK, scoring, steering z_16646 = 0.
- Repo abhi GitHub par nahi bana: account bsf1703846 par 5 repos hain, event-sae-demo nahi hai, create_repository 403 'Resource not accessible by integration' se fail hua. Repo manually banana parega github.com/new par, ya GitHub App install karni paregi.
- LinkedIn post abhi nahi kiya gaya — sirf draft hai, professor tag placeholder hai, tumhare yes ka wait hai.

## Abhi baki kya hai
1. Professor confirm karo — Levine ya Metaxas, ya dono.
2. GitHub repo manually banao (event-sae-demo, public), phir files push kar dunga.
3. LinkedIn post approve karo, phir post karunga.
4. Kang pilot (CLIP zero-shot) Event-SAE ke baad shuru karna hai, aur Kang thank-you email tumhari approval se bhejna hai.
5. Mao Piper voice decision — live se pehle ya text-only first live.
6. Dartmouth fee waiver form bhara ya nahi, ye confirm karo.
