insts <- c("AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD")

# Run only once, to create daily candles
for (c1 in insts) {
  save_daily(c1)
  for (c2 in insts) {
    pair <- paste0(c1, c2)
    f <- paste0(pair, "2020M5.rds")
    if (file.exists(f)) save_daily(pair)
  }
}

save_daily <- function(instrument) {
  d1 <- to.daily(readRDS(paste0(instrument, "2020M5.rds")))
  colnames(d1) <- c("Open", "High", "Low", "Close")
  saveRDS(d1, file=paste0(".\\Daily\\", instrument, '.rds'))
}

# Once daily's are already saved
currency_vol <- vector()
pair_vol <- vector()
find_vol <- function(instrument) {
  d1 <- readRDS(paste0(".\\Daily\\", instrument, '.rds'))
  return((max(d1$High) / min(d1$Low)) * 100 - 100)
}

for (c1 in insts) {
  currency_vol <- rbind(currency_vol, find_vol(c1))
  for (c2 in insts) {
    pair <- paste0(c1, c2)
    f <- paste0(".\\Daily\\", pair, '.rds')
    if (file.exists(f)) {
      pair_vol <- rbind(pair_vol, find_vol(pair))
    }
  }
}

d1 <- data.frame(pair_vol, "Currency pair")
d2 <- data.frame(currency_vol, "Currency")
colnames(d1) <- c("Volatility", "Instrument")
colnames(d2) <- c("Volatility", "Instrument")
st_devs <- rbind(d1, d2)
st_devs$Instrument <-
  factor(st_devs$Instrument, levels=c("Currency pair", "Currency"))
g <- ggplot(st_devs, aes(x=Instrument, Volatility)) +
  geom_boxplot() + xlab(NULL)
boxdata <- ggplot_build(g)$data[[1]]
q1 <- boxdata[1,2] # first quartile of currency pair
q3 <- boxdata[2,4] # third quartile of currency
g + geom_hline(yintercept = (q1+q3) / 2, linetype="dashed", color = "red")
