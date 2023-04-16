## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, fig.align = "left")
options(ggrepel.max.overlaps = Inf)

## ---- eval = FALSE------------------------------------------------------------
#  install.packages('RcppML')                     # install CRAN version
#  # devtools::install_github("zdebruine/RcppML") # compile dev version

## ---- message = FALSE, warning = FALSE----------------------------------------
library(RcppML)
library(ggplot2)
library(cowplot)
library(viridis)
library(ggrepel)
library(uwot)

## -----------------------------------------------------------------------------
data(hawaiibirds)
hawaiibirds$counts[1:4, 1:4]

## -----------------------------------------------------------------------------
head(hawaiibirds$metadata_h)

## -----------------------------------------------------------------------------
head(hawaiibirds$metadata_w)

## ---- fig.width = 4, fig.height = 3, eval = FALSE-----------------------------
#  plot(crossValidate(hawaiibirds$counts, k = c(1:20), reps = 3)) + scale_y_continuous(trans = "log10")

## ---- results = "hide"--------------------------------------------------------
model <- nmf(hawaiibirds$counts, k = 15, seed = 1:3, tol = 1e-6)

## -----------------------------------------------------------------------------
model

## ---- warning = FALSE, message = FALSE, fig.width = 4, fig.height = 4---------
plots <- list()
for(i in 1:4){
  df <- data.frame(
    "lat" = hawaiibirds$metadata_h$lat,
    "lng" = hawaiibirds$metadata_h$lng,
    "nmf_factor" = model$h[i, ])
  plots[[i]] <- ggplot(df, aes(x = lng, y = lat, color = nmf_factor)) +
    geom_point() +
    scale_color_viridis(option = "B") +
    theme_void() +
    theme(legend.position = "none", plot.title = element_text(hjust = 0.5)) + 
    ggtitle(paste0("Factor ", i))
}
plot_grid(plotlist = plots, nrow = 2)

## ---- fig.width = 4, fig.height = 3-------------------------------------------
plot(summary(model, group_by = hawaiibirds$metadata_h$island, stat = "mean"))

## ---- fig.width = 4, fig.height = 3-------------------------------------------
plot(summary(model, group_by = hawaiibirds$metadata_w$type, stat = "mean"))

## ---- message = FALSE, warning = FALSE, fig.width = 5.5, fig.height = 4-------
biplot(model, factors = c(2, 3), matrix = "w", group_by = hawaiibirds$metadata_w$type) + 
  scale_y_continuous(trans = "sqrt") + 
  scale_x_continuous(trans = "sqrt") +
  geom_text_repel(size = 2.5, seed = 123, max.overlaps = 15) +
  theme(aspect.ratio = 1)

## ---- warning = FALSE, message = FALSE, fig.width = 8, fig.height = 2.5-------
set.seed(123)
umap <- data.frame(uwot::umap(model$w))
umap$taxon <- hawaiibirds$metadata_w$type
umap$status <- hawaiibirds$metadata_w$status
plot_grid(
  ggplot(umap, aes(x = umap[,1], y = umap[,2], color = taxon)) +
    geom_point() + theme_void(),
  ggplot(umap, aes(x = umap[,1], y = umap[,2], color = status)) +
    geom_point() + theme_void(),
  nrow = 1
)

## ---- warning = FALSE, message = FALSE, fig.width = 4, fig.height = 2.5-------
set.seed(123)
umap <- data.frame(uwot::umap(t(model$h), metric = "cosine"))
umap$group <- hawaiibirds$metadata_h$island
ggplot(umap, aes(x = umap[,1], y = umap[,2], color = group)) +
  geom_point() + theme_void() + theme(aspect.ratio = 1)

## -----------------------------------------------------------------------------
ggplot(data.frame("value" = model$w["Palila", ], "factor" = 1:ncol(model$w)), aes(factor, value)) + 
  geom_point() + 
  theme_classic() + 
  theme(aspect.ratio = 1) + 
  labs(x = "NMF factor", y = "Palila weight in NMF factor")

df <- data.frame("value" = model$w[, which.max(model$w["Palila", ])])
df$status <- hawaiibirds$metadata_w$status
df <- df[order(-df$value), ]
df <- df[df$value > 0.001, ]
df

## -----------------------------------------------------------------------------
perching_birds <- hawaiibirds$metadata_w$species[hawaiibirds$metadata_w$type == "perching birds"]
df[which(rownames(df) %in% perching_birds & df$status == "introduced"), ]

