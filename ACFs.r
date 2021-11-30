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
ggc <- ggplot()
ggp <- ggplot()
i <- 0
v <- vector()
find_vol <- function(instrument) {
  d1 <- readRDS(paste0(".\\Daily\\", instrument, '.rds'))
  # return((max(d1$High) / min(d1$Low)) * 100 - 100)
  # return(((d1$Close[dim(d1)[2], 4] - d1$Open[1, 1]) / d1$Open[1]) * 100)
  # return(((d1[dim(d1)[2], 4] - d1[1, 1]) / d1[1]) * 100)
  returns <- d1$Close / stats::lag(d1$Close) - 1
  mean_return <- mean(abs(returns*100), na.rm=TRUE)
  ggAcf(d1$Close, lag.max = 10)
  # returns <- sd(returns*100, na.rm=TRUE)
  # return(list("Returns"=returns, "Mean"=mean_return, "ACF"=g))
  # return(g)
}

for (c1 in insts) {
  i <- i + 1
  # currency_vol <- rbind(currency_vol, find_vol(c1))
  # g <- find_vol(c1)$ACF
  find_vol(c1)
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






# v[i] <- g
# ggc <- ggc + g
# for (c2 in insts) {
#   pair <- paste0(c1, c2)
#   f <- paste0(".\\Daily\\", pair, '.rds')
#   if (file.exists(f)) {
#     ggp <- ggp + find_vol(pair)$ACF
#     # pair_vol <- rbind(pair_vol, find_vol(pair))
#   }
# }
