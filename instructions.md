Progetto: filtering on graphs using the Lanczos method Lo scopo del progetto è
implementare e testare il metodo di Lanczos per il calcolo di quantità della
forma f(A)b dove A è la matrice Laplaciano di un grafo.

Legga il lavoro [1], che descrive l'applicazione del calcolo di funzioni di
matrici a filtri su grafi e si implementi il metodo di Lanczos per f(A)b (vicino
parente del metodo di Arnoldi visto a lezione). Per testare l'algoritmo scarichi
il toolbox GSP reperibile su [2] che permette di generare matrici di adiacenza
di grafi random di tipo "Erdos-Reny" e "sensor" (si veda la documentazione del
toolbox https://epfl-lts2.github.io/gspbox-html/doc/graphs/).

Quindi, si effettuino i seguenti tests:

1. Si ripeta il test corrispondente ad Example 1 dell'articolo limitandosi al
   metodo di Lanczos (no Chebyshev) e utilizzando come funzione g(t) = sin(0.5π
   cos(πt)2) * \chi_{[-0.5, 0.5]}.
2. Si generino grafi di di Erdos-Reny di grandezza crescente (ad esempio 250,
   500,1000, 2000, 4000) e parametro p = 0.04; quindi si misuri il tempo
   computazionale del metodo di Lanczos utilizzando come soglia per il criterio
   d'arresto epsilon = 10^-2 (o una soglia a scelta).
3. Esperimento analogo a 2) ma si tenga la dimensione fissata a n=1000 e si
   aumenti il parametro p, ad esempio p = 0.01, 0.02, 0.04, 0.08, 0.16, 0.32.

In tutti gli esperimenti si scelga come vettore b un vettore random di norma
unitaria.

Si mostrino i risultati di 2) e 3) in un grafico in scala loglog al fine di
determinare la dipenza della complessità computazionale dai parametri n e p.

Suggerimento: Nel caso si osservi una convergenza troppo lenta dell'algoritmo si
controlli che il problema non sia nell'ortogonalità della base generata da
Lanczos. Nel caso si sostituisca Lanczos con il metodo di Arnoldi per f(A)b, che
riortogonalizza ad ogni passo i vettori generati rispetto a tutti i precedenti.

Una volta conclusa la sperimentazione, scriva un report che descriva l’algoritmo
implementato, e i risultati ottenuti nei test numerici; includa anche il codice,
e lo consegni inviandolo a me e a Robol; oltre ai docenti anche Alberto Bucci,
che ha l’incarico di tutor per il corso, è disponibile per assistenza e domande
durante lo svolgimento del progetto.

[1] Susnjara et al., Accelerated filtering on graphs using Lanczos method. [2]
https://epfl-lts2.github.io/gspbox-html/
