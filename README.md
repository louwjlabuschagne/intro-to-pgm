# Intro To PGM

## Resource

https://mbmlbook.com/
## Setup

Ubuntu:

```bash
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.deb

sudo apt install -y dotnet-sdk-3.1
sudo apt install -y aspnetcore-runtime-3.1
sudo apt install graphviz
```

## Examples 

+ <a href="./notebooks/coin-toss.ipynb">Coin Toss</a>
+ <a href="./notebooks/iris-bayes-point-classifier.ipynb">Bayes Point Machine</a>

