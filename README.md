# better-netl

A Python 3 fork of Shraey Bhatia's NETL approach.

> [!NOTE]  
> This README is under construction. If you are looking for the original README, check out [Shraey Bhatias repository](https://github.com/sb1992/NETL-Automatic-Topic-Labelling-).

## Motivation & Goals

The original NETL approach is a great idea, but its last code update is dated back to 2016. Since then, Python 2 was marked as deprecated and further changes in the `gensim` library etc. have made this code unusable. This fork aims to provide a Python 3 compatible version of the original NETL approach. We also use this opportunity to improve the implementation itself to make it, again, more usable.

- [x] Python 3 compatibility
- [ ] Code restructure and readability improvements
- [ ] Use HuggingFace's `datasets` library for loading the Wikipedia data
