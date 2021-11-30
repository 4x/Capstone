library(ggplot2)
library(ggtext)
library(xts)

base.currency <- "NZD"
quote.currency <- "USD"
pair <- paste0(base.currency, quote.currency)

# On first run only:
# nzdusd <- readRDS(paste0(base.currency, ".rds"))$Bid /
#           readRDS(paste0(quote.currency, ".rds"))$Bid
# saveRDS(nzdusd, file="SyntheticNZDUSDticks.rds")

# nzdusd 	<- na.locf(na.locf(
#   merge(readRDS(paste0(".\\Daily\\", pair, '.rds'))[,4],
#         to.daily(readRDS("SyntheticNZDUSDticks.rds"))[,4])), fromLast = TRUE)
# colnames(nzdusd) <- c("Actual", "Synthetic")
# saveRDS(nzdusd, file=(paste0(".\\Daily\\", "ActualSyntheticNZDUSD.rds")))
nzdusd <- readRDS(paste0(".\\Daily\\", "ActualSyntheticNZDUSD.rds"))

d <- data.frame(
  day = as.Date(index(nzdusd)),
  Actual <- nzdusd[,1],
  Synthetic <- nzdusd[,2]
)

daily_close <- ggplot(d) +
  geom_line(mapping=aes(x=day, y=Actual, color="Actual"),
            linetype="dashed", size=1) +
  geom_line(mapping=aes(x=day, y=Synthetic, color="Synthetic"),
            linetype="dotted", size=1) +
  scale_color_manual(values
                     = c('Actual' = 'aquamarine', 'Synthetic' = 'seagreen')) +
  labs(x = NULL, y = "NZD/USD", color = "Market",
    title = "Actual vs synthetic daily close for 2020",
    subtitle = paste0("*Synthetic* data represented as dots ",
                      "to visually separate from *actual* market data"))
daily_close + theme(plot.subtitle = ggtext::element_markdown())
