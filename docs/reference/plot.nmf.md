# Plot NMF Training History and Diagnostics

TensorBoard-style visualization of NMF training dynamics, convergence
analysis, and factor diagnostics. Provides multiple plot types for
comprehensive model analysis.

## Usage

``` r
# S3 method for class 'nmf'
plot(
  x,
  type = c("loss", "convergence", "regularization", "sparsity"),
  smooth = TRUE,
  span = 0.3,
  log_scale = FALSE,
  interactive = FALSE,
  theme = "classic",
  ...
)
```

## Arguments

- x:

  object of class "nmf"

- type:

  plot type: - "loss": Loss components over iterations (default) -
  "convergence": Log-scale loss convergence - "regularization":
  Regularization penalty contributions - "sparsity": Factor sparsity
  patterns

- smooth:

  apply smoothing (LOESS) for noisy curves (default TRUE)

- span:

  smoothing span for LOESS (default 0.3)

- log_scale:

  use log scale for y-axis (default FALSE, auto TRUE for "convergence")

- interactive:

  create interactive plotly plot (default FALSE)

- theme:

  ggplot2 theme: "classic", "minimal", "dark" (default "classic")

- ...:

  additional arguments passed to specific plotting functions

## Value

ggplot2 or plotly object

## See also

[`nmf`](https://zdebruine.github.io/RcppML/reference/nmf.md),
[`compare_nmf`](https://zdebruine.github.io/RcppML/reference/compare_nmf.md)

## Examples

``` r
# \donttest{
# Basic loss plot
model <- nmf(hawaiibirds, k = 10)
plot(model)


# Convergence analysis
plot(model, type = "convergence")


# Interactive plot
plot(model, type = "loss", interactive = TRUE)

{"x":{"data":[{"x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"y":[1805.7626953125,1188.39013671875,1037.233642578125,961.3623046875,908.896240234375,865.43701171875,839.4365234375,826.398193359375,819.90771484375,816.433349609375,814.599609375,813.564208984375,812.95166015625,812.6552734375,812.545654296875,812.508056640625,812.5302734375,812.58935546875,812.6640625,812.7373046875],"text":["iteration:  1<br />loss: 1805.7627<br />loss_type: train_loss","iteration:  2<br />loss: 1188.3901<br />loss_type: train_loss","iteration:  3<br />loss: 1037.2336<br />loss_type: train_loss","iteration:  4<br />loss:  961.3623<br />loss_type: train_loss","iteration:  5<br />loss:  908.8962<br />loss_type: train_loss","iteration:  6<br />loss:  865.4370<br />loss_type: train_loss","iteration:  7<br />loss:  839.4365<br />loss_type: train_loss","iteration:  8<br />loss:  826.3982<br />loss_type: train_loss","iteration:  9<br />loss:  819.9077<br />loss_type: train_loss","iteration: 10<br />loss:  816.4333<br />loss_type: train_loss","iteration: 11<br />loss:  814.5996<br />loss_type: train_loss","iteration: 12<br />loss:  813.5642<br />loss_type: train_loss","iteration: 13<br />loss:  812.9517<br />loss_type: train_loss","iteration: 14<br />loss:  812.6553<br />loss_type: train_loss","iteration: 15<br />loss:  812.5457<br />loss_type: train_loss","iteration: 16<br />loss:  812.5081<br />loss_type: train_loss","iteration: 17<br />loss:  812.5303<br />loss_type: train_loss","iteration: 18<br />loss:  812.5894<br />loss_type: train_loss","iteration: 19<br />loss:  812.6641<br />loss_type: train_loss","iteration: 20<br />loss:  812.7373<br />loss_type: train_loss"],"type":"scatter","mode":"lines","line":{"width":1.8897637795275593,"color":"rgba(31,119,180,0.3)","dash":"solid"},"hoveron":"points","name":"train_loss","legendgroup":"train_loss","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"y":[1773.1446424886115,1278.1064722546862,1011.0100645087349,958.46333806094754,907.12890234966483,866.43850003961529,839.58484801770646,826.17524385988531,819.74094174478364,816.39210952240228,814.56350541607526,813.54341899547524,812.95341625885453,812.65414046807769,812.53772351091891,812.50888748486796,812.53014125773416,812.58903446074703,812.65839326982234,812.73944280161709],"text":["iteration:  1<br />loss_smooth: 1773.1446<br />loss_type: train_loss","iteration:  2<br />loss_smooth: 1278.1065<br />loss_type: train_loss","iteration:  3<br />loss_smooth: 1011.0101<br />loss_type: train_loss","iteration:  4<br />loss_smooth:  958.4633<br />loss_type: train_loss","iteration:  5<br />loss_smooth:  907.1289<br />loss_type: train_loss","iteration:  6<br />loss_smooth:  866.4385<br />loss_type: train_loss","iteration:  7<br />loss_smooth:  839.5848<br />loss_type: train_loss","iteration:  8<br />loss_smooth:  826.1752<br />loss_type: train_loss","iteration:  9<br />loss_smooth:  819.7409<br />loss_type: train_loss","iteration: 10<br />loss_smooth:  816.3921<br />loss_type: train_loss","iteration: 11<br />loss_smooth:  814.5635<br />loss_type: train_loss","iteration: 12<br />loss_smooth:  813.5434<br />loss_type: train_loss","iteration: 13<br />loss_smooth:  812.9534<br />loss_type: train_loss","iteration: 14<br />loss_smooth:  812.6541<br />loss_type: train_loss","iteration: 15<br />loss_smooth:  812.5377<br />loss_type: train_loss","iteration: 16<br />loss_smooth:  812.5089<br />loss_type: train_loss","iteration: 17<br />loss_smooth:  812.5301<br />loss_type: train_loss","iteration: 18<br />loss_smooth:  812.5890<br />loss_type: train_loss","iteration: 19<br />loss_smooth:  812.6584<br />loss_type: train_loss","iteration: 20<br />loss_smooth:  812.7394<br />loss_type: train_loss"],"type":"scatter","mode":"lines","line":{"width":3.7795275590551185,"color":"rgba(31,119,180,1)","dash":"solid"},"hoveron":"points","name":"train_loss","legendgroup":"train_loss","showlegend":false,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null}],"layout":{"margin":{"t":41.90286425902864,"r":7.3059360730593621,"b":37.260273972602747,"l":48.949771689497723},"plot_bgcolor":"rgba(255,255,255,1)","paper_bgcolor":"rgba(255,255,255,1)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724},"title":{"text":"<b> NMF Training History <\/b>","font":{"color":"rgba(0,0,0,1)","family":"","size":18.596928185969279},"x":0,"xref":"paper"},"xaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[0.049999999999999933,20.949999999999999],"tickmode":"array","ticktext":["5","10","15","20"],"tickvals":[5,10,15,20],"categoryorder":"array","categoryarray":["5","10","15","20"],"nticks":null,"ticks":"outside","tickcolor":"rgba(0,0,0,1)","ticklen":3.6529680365296811,"tickwidth":0,"showticklabels":true,"tickfont":{"color":"rgba(0,0,0,1)","family":"","size":11.68949771689498},"tickangle":-0,"showline":true,"linecolor":"rgba(0,0,0,1)","linewidth":0,"showgrid":false,"gridcolor":null,"gridwidth":0,"zeroline":false,"anchor":"y","title":{"text":"Iteration","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724}},"hoverformat":".2f"},"yaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[762.8453247070313,1855.4254272460937],"tickmode":"array","ticktext":["1000","1250","1500","1750"],"tickvals":[1000,1250,1500,1750],"categoryorder":"array","categoryarray":["1000","1250","1500","1750"],"nticks":null,"ticks":"outside","tickcolor":"rgba(0,0,0,1)","ticklen":3.6529680365296811,"tickwidth":0,"showticklabels":true,"tickfont":{"color":"rgba(0,0,0,1)","family":"","size":11.68949771689498},"tickangle":-0,"showline":true,"linecolor":"rgba(0,0,0,1)","linewidth":0,"showgrid":false,"gridcolor":null,"gridwidth":0,"zeroline":false,"anchor":"x","title":{"text":"Loss","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724}},"hoverformat":".2f"},"shapes":[{"type":"rect","fillcolor":null,"line":{"color":null,"width":0,"linetype":[]},"yref":"paper","xref":"paper","layer":"below","x0":0,"x1":1,"y0":0,"y1":1}],"showlegend":true,"legend":{"bgcolor":"rgba(255,255,255,1)","bordercolor":"transparent","borderwidth":0,"font":{"color":"rgba(0,0,0,1)","family":"","size":11.68949771689498},"title":{"text":"Loss Type","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724}}},"hovermode":"closest","barmode":"relative"},"config":{"doubleClick":"reset","modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"source":"A","attrs":{"33cf1e75815bb":{"x":{},"y":{},"colour":{},"type":"scatter"},"33cf1e4716a5af":{"x":{},"y":{},"colour":{}}},"cur_data":"33cf1e75815bb","visdat":{"33cf1e75815bb":["function (y) ","x"],"33cf1e4716a5af":["function (y) ","x"]},"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}
# Compare multiple runs
models <- replicate(5, nmf(hawaiibirds, k = 10), simplify = FALSE)
plot(models[[1]], type = "sparsity")

# }
```
