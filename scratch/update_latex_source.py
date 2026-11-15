from pathlib import Path
import re

src = Path('docs/research_paper_ieee_updated.tex')
text = src.read_text(encoding='utf-8')
abstract = r'''\begin{abstract}
In competitive multi-agent reinforcement learning (MARL), unanchored self-play often develops cycles as each agent adapts to its opponent, a pattern known as the Red Queen effect. This shifting competition can also cause catastrophic forgetting, where a policy learns to counter recent behavior but loses previously learned routes. We investigate these problems in the Three Bridges Arena, an asymmetric zero-sum gridworld with three bridge crossings and one Defender, which cannot guard every crossing at once. We propose a symmetric Dual-Archive Co-Evolution approach that trains current policies against one another and against archived earlier policies. It also uses discounted potential-based reward shaping with a discount factor of 0.99, designed to preserve optimal policies under the policy-invariance result of Ng et al. (1999). Across 150 paired evaluation seeds, naive self-play showed route collapse: 46.0 percent North bridge traversal compared with 16.1 percent South, 17.3 percent unforced navigation timeouts, and an exploitability score of 0.160. Dual-Archive co-evolution produced more balanced use of the three corridors, with Shannon route entropy of 1.567 bits, 98.9 percent of the theoretical maximum of 1.585 bits, and no timeouts. It also reduced minimax exploitability to 0.140 and improved the worst-case security level from -0.207 to -0.187. Pooled win rates did not differ significantly between the methods: the difference was -0.6 percent, with a p-value of 0.771. These results suggest that pooled win rates alone can hide strategic robustness differences, while route entropy and empirical minimax exploitability provide additional measures.
\end{abstract}'''
text, n = re.subn(r'\\begin\{abstract\}.*?\\end\{abstract\}', lambda _match: abstract, text, count=1, flags=re.S)
assert n == 1
text = text.replace(r'\cite{wellman2006methods}', r'\cite{lanctot2017unified}')
text = text.replace(r'Prioritized fictitious play \cite{lanctot2017unified} or regret-weighted sampling', 'Prioritized fictitious play or regret-weighted sampling')
text = text.replace(r'clustering techniques \cite{balduzzi2019open}', 'clustering techniques')
text = text.replace(r'\begin{thebibliography}{0}', r'\begin{thebibliography}{15}')
addition = r'''
\bibitem{mccloskey1989catastrophic} M. McCloskey and N. J. Cohen, ``Catastrophic interference in connectionist networks: The sequential learning problem,'' in \textit{Psychology of Learning and Motivation}, vol. 24, pp. 109--165, 1989.
\bibitem{silver2016mastering} D. Silver \textit{et al.}, ``Mastering the game of Go with deep neural networks and tree search,'' \textit{Nature}, vol. 529, pp. 484--489, 2016.
\bibitem{vinyals2019alphastar} O. Vinyals \textit{et al.}, ``Grandmaster level in StarCraft II using multi-agent reinforcement learning,'' \textit{Nature}, vol. 575, pp. 350--354, 2019.
\bibitem{berner2019dota} OpenAI \textit{et al.}, ``Dota 2 with large scale deep reinforcement learning,'' Dec. 2019.
\bibitem{balduzzi2019open} D. Balduzzi \textit{et al.}, ``Open-ended learning in symmetric zero-sum games,'' in \textit{Proc. 36th Int. Conf. Machine Learning}, PMLR, vol. 97, pp. 434--443, 2019.
\bibitem{devlin2011theoretical} S. Devlin and D. Kudenko, ``Theoretical considerations of potential-based reward shaping for multi-agent systems,'' in \textit{Proc. 10th Int. Conf. Autonomous Agents and Multiagent Systems}, 2011, pp. 225--232.
\bibitem{tesauro1995temporal} G. Tesauro, ``Temporal difference learning and TD-Gammon,'' \textit{Communications of the ACM}, vol. 38, no. 3, pp. 58--68, 1995.
\bibitem{brown1951iterative} G. W. Brown, ``Iterative solution of games by fictitious play,'' in \textit{Activity Analysis of Production and Allocation}, T. C. Koopmans, Ed. New York, NY, USA: Wiley, 1951.
\bibitem{mcmahan2003planning} H. B. McMahan, G. J. Gordon, and A. Blum, ``Planning in the presence of cost functions controlled by an adversary,'' in \textit{Proc. 20th Int. Conf. Machine Learning}, 2003, pp. 536--543.
\bibitem{lanctot2017unified} M. Lanctot \textit{et al.}, ``A unified game-theoretic approach to multiagent reinforcement learning,'' in \textit{Advances in Neural Information Processing Systems}, vol. 30, 2017.
\bibitem{heinrich2016deep} J. Heinrich and D. Silver, ``Deep reinforcement learning from self-play in imperfect-information games,'' arXiv:1603.01121, 2016.
\bibitem{von1944theory} J. von Neumann and O. Morgenstern, \textit{Theory of Games and Economic Behavior}. Princeton, NJ, USA: Princeton University Press, 1944.'''
text = text.replace(r'\end{thebibliography}', addition + '\n\\end{thebibliography}')
src.write_text(text, encoding='utf-8')
print(f'Updated {src}')
