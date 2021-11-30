insts <- c("AUD", "NZD", "EUR", "GBP", "CAD", "CHF", "JPY", "USD")
df <- data.frame(Instrument=character(), Size=integer(), Type=character())

merge_currency <- function(instrument, df) {
  pc <- "Currency"
  if (nchar(instrument) > 3) pc <- "Currency pair"
  f <- paste0(".\\M1\\", instrument, '2020M1.rds')
  d <- list(Instrument=instrument, Size=file.info(f)$size / 2^20, Type=pc)
  df <- rbind(d, df, stringsAsFactors=FALSE)
  return(df)
}

for (c1 in insts) {
  df <- merge_currency(c1, df)
  for (c2 in insts) {
    c12 <- paste0(c1, c2)
    f <- paste0(".\\M1\\", c12, '2020M1.rds')
    if (file.exists(f)) {
      df <- merge_currency(c12, df)
    }
  }
}

df$Type <- factor(df$Type, levels=c("Currency pair", "Currency"))

library(ggplot2)
ggplot(df, aes(fill=Instrument, y=Size, x=Type)) + 
  geom_bar(position="stack", stat="identity") +
  xlab(NULL) + ylab("File size [MB]")
