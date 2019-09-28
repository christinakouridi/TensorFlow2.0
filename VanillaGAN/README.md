This is an implementation of a Generative Adversarial Network from Ian Goodfellow's paper (\url{https://arxiv.org/pdf/1406.2661.pdf}), where the Generator and Discriminator are described by multilayer perceptrons. Several empirical techniques have been implemented to improve stability: <br>
\begin{itemize}
\item One-sided label smoothing: The discriminator has been penalised when the prediction for any real images goes beyond 0.9, by setting our target label value to be 0.9 instead of 1.0. 
\item The Adam optimiser is used instead of SGD with momentum, to accelerate convergence and prevent oscillations in local minima
\item Noisy images are spherical instead of linear, by sampling from gaussian rather than a uniform distribution
\item The image pixels are scaled between -1 and 1, and tanh is used as the output layer for the generator
\item Batch normalisation is used to further stabilise training
\end{itemize}
For more tricks, visit \url{https://github.com/soumith/ganhacks}
