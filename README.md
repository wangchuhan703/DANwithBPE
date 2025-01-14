# DANwithBPE
Introduction:  A DAN (Deep Averaging Network) network implementation using the subword tokenization method -- BPE (Byte Pair Encoding).
# **Text sentiment analysis on DAN models**

## **How to run the files:**

****Part 1:** DAN model**

**1a)**

Run `python main.py --model DAN`



**1b)** 

Find line 190-195 in 'main.py'  -> change to '1b' ->
Run `python main.py --model DAN`

****Part 2:** BPE**

**2a)**

First, run `python BPE.py` to generate subword dictionary 'subword_vocab.txt'.

Second, run `python main.py --model SUBWORDDAN`.
