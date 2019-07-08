# Text-Generation

Les codes fournis ici permettent d'utiliser des modèles de "Language Modeling" pour réaliser de la génération de textes.
L'implémentation a été réalisée sur base de ce papier : [Language Models are Unsupervised Multitask Learners](https://paperswithcode.com/paper/language-models-are-unsupervised-multitask).
Elle reprend en partie les codes disponibles sur le github d'[HuggingFace](https://github.com/huggingface/pytorch-pretrained-BERT) ainsi que les modèles pré-entrainés.

## Requirements
<ul>
  <li>Python 3.7</li>
  <li>Torch 1.1.0</li>
  <li>Torchvision</li>
  <li>tqdm</li>
  <li>regex</li>
  <li>ftfy</li>
  <li>pytorch-pretrained-bert</li>
</ul>

## Modèles disponibles pré-entrainés
4 Modèles ont été testés : 
<ol>
  <li> BERT </li>
  <li> OpenAI-GPT </li>
  <li> OpenAI-GPT2 </li>
  <li> Transformer-XL </li>
</ol>

Le code pour BERT ne permet de faire que du masquage d'un seul moment. Le modèle propose alors une série de mots de remplacement parmi ceux qu'il connait. En voici un exemple :

<pre><code>Original: All my friends were coming at the party.
Masked: all my [MASK] were coming at the party .
Predicted token: ['parents']
Other options:
['friends']
['kids']
['people']
['they']
['mom']
['classes']
['girls']
['thoughts']
['things']
['own']
</code></pre>

Des implémentations rapides des modèles OpenAI-GPT et OpenAI-GPT2 donnent des résultats peu intéressants. Par exemple, sur le début de phrase "Give this a little try" pour OpenAI-GPT: 
<pre><code>give this a little try . " 
 " i 'm not sure i can . " 
 " you can . " 
 " i do n't know . " 
 " you can . " 
 " i do n't know . "
 </code></pre>
 
 Et sur "Maybe this will work" pour OpenAI-GPT2 : 
 <pre><code>Maybe this will work for you.
The first thing you need to do is to create a new file called "config.json" in your project's root directory.
In this file, you'll need to add the following line to your .bashrc :
{ "name": "config.json", "version": "1.0", "version_id": "1", "version_name": "config.json", "version_name_id": "1", "version_name_name": "config.json", "version_name_name_id": "
 </code></pre>
 
Le code d'exemple de génération de texte fourni par HuggingFace pour OpenAI-GPT2 est plus probant:
<pre><code>Sample N°1 : It wasn't in her power to approve or deny cleaning up afterwards. That knowledge had a kaleidoscope of charms, and could supplement or destroy her desires. She was suspiciously devoted once the compound ...
Sample N°2 : "We can speak normally only because we are members of the Cyprus international perspective and as such have the backpedaling responsibility to ensure the good Standing Committee was unable at its minimum to get a result on ... 
Sample N°3 : NOT AN EXCELLENT DOUBLE AND SURFER README HERE.
Updated 8:30 pm EST, 02:12 pm (CST)
A Rogue Shadow full RA</code></pre>

De même que sa réimplémentation, disponible ici, pour OpenAI-GPT : 
<pre><code>Sample N°1 : if there was no guarantee that their family needs something to feed in return. but the only thing that is particularly disconcerting is that they are at some point in their life when they keep the place they love their oldest child unattended any longer.
Sample N°2 : and this guy had shot my friend. the premier fellow took a drink, wiped his mouth on his sleeve, and then proceeded to make a scene, noisily retching into his handkerchief. at that point, we were elated with the dejected approach to telling the</code></pre>

Enfin, il reste le modèle "Transformer-XL" qui a été entrainé sur des pages Wikipédia.
Avec le texte "First world war", il produit par exemple : 
<pre><code>= I Just Can 't Stop Believing You 're Getting to Stop = an The Star-Ledger-Enquirer article about him titled " The Sun Rises About The Stars ". 
= = Early career and life = = 

Alan Moore was born in Manchester, England, to parents George Arthur Moore and Edith Moore. He studied English at St John's College, Cambridge, and served first in the Royal Marines during World War I, and then in the Royal Air Force for four years. While stationed at Yeovilton, he met Edith Moore, who later recounted their initial meeting: " I went and sat down to read novels, and said to her, ' I would like you to stay behind for two days, take my exams, and do anything else. ' But the last time I returned the next day, Edith wrote down a second thing she said, ' I 'd like you to leave me alone. ' " She eventually made it to England, where she married Moore and they had two children 
</pre></code>

## Fine-tuning pour OpenAI-GPT
Il est possible de spécialiser un modèle générique pour qu'il produise des échantillons d'un style plus spécifique. Ainsi, en utilisant la commande suivante : 
<pre><code>python ine_tuning_openai.py --do_train --output_dir train_clarke --train_dataset FineTuning-example.txt</pre></code>

Le modèle généré sera capable de produire des extraits de texte ressemblant au premier tome de l'Odyssey de Clarke. Les marqueurs "\_end\_" indiquent des changements de paragraphe. 
<pre><code>INPUT : Dr . Floyd was really angry because
  OUTPUT : the party must have held up late. he opened the hatch briefly, and then, with more caution than confidence, stepped out into the night he had once inhabited, back across the crater of earth, and back into solid space. _end_... on the dark sides of the mountains and valleys... ( this had happened again, more times than he could count. ) _end__end_... _end_that was improvicomspheres of the satellites, staring out their movements, until they prickled the worst of glory and then, as the mist moved across the sky. _end__end__end__end_one hour, gradually transparrends the void, then continued along the air. _end_even cautiously improving its dominion over the wild without resistance, until the rhythms crept out upon the black emptiness. for this did not allow the loneliness of time drove itself to eternity. that had been coded with turbulent infinities of identity. _end_day by night, before descending into complacment, until it had become nothing but permanent. _end__end_... 
 his ears, listening to the stars beyond his senses. _end_nuclear freeze ; in concentration. only the world, until the mini suns passed across the empty self, and as it had found its limits. he did not know why he loosed itself off its rhythm. but the night had the same </pre></code>
 <pre><code> INPUT : I heard some voices with
 OUTPUT : difficulty, as if the body was being carried around by a tremendous washing machine, pumping out vast outputs of hot air. bowman watched the machine in fascinated fascination ; he felt as if he could breathe it. _end_... the laws were the same ; yet he had never truly known them. _end_even more baffling, as the breathing was hurled aside like lightning through the darkness, was one of infinite possibilities. _end_... the what else was deeper and deeper, the ceaseless grind of drifting impulses... _end_... the long - still discovery immersed him, discovery's first mortifying eruptions of bowels. _end_man discovered resistance : sudden bombing, peredogs - possible the effect of infinite rapture. _end_... and then, like malice ; then human habitability, acute exploration. then phansis, isolation ; and ultimately, _end__end_the watch after victory over the planet earth, voyeur gs - aching exercise retribush - disappointment the retreat ; retreat from achievement, vigilance toward rock - face this and finally, collapse into silence. _end__end_tanks, retreat from deep seas of fury... as lonely and eternal. _end__end_kill the dream. they might die in utter silence ; crystal canoes filled with fire. _end_... retreat, retreat - this world empty. _end_of dullife... the world of dreams ;</pre></code>
 
## Génération en français
Concernant de la génération de texte en français, il n'existe pas ou peu de base de données. En tout cas, les modèles ici présentés n'ont pas d'équivalent français. Dans le cas où un utilisateur souhaiterait donc faire de la génération en français, il faudrait avant tout ré-entrainer un modèle.
