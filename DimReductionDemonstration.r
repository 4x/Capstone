insts <- c("AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD")

currency <- xts()
pair <- xts()
merge_currency <- function(instrument) {
  r <- readRDS(paste0(".\\M1\\", instrument, '2020M1.rds'))[,4]
  colnames(r) <- instrument
  return(r)
}

for (c1 in insts) {
  currency <- na.locf(na.locf(merge(currency, merge_currency(c1), join="right"),
                              fromLast = TRUE))
  for (c2 in insts) {
    c12 <- paste0(c1, c2)
    f <- paste0(".\\M1\\", c12, '2020M1.rds')
    if (file.exists(f)) {
      pair <- merge(pair, merge_currency(c12), join="right")
    }
  }
}
pair[endpoints(pair, on="quarters"),]
print(dim(pair)[2])
currency[endpoints(currency, on="quarters"),]
print(dim(currency)[2])