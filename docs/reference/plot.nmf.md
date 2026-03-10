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
model <- nmf(hawaiibirds, k = 10, track_loss_history = TRUE)
plot(model)


# Convergence analysis
plot(model, type = "convergence")


# Interactive plot
plot(model, type = "loss", interactive = TRUE)

{"x":{"data":[{"x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"y":[1805.76171875,1188.38916015625,1037.232177734375,961.3623046875,908.895751953125,865.4375,839.4365234375,826.398681640625,819.90869140625,816.433349609375,814.5986328125,813.563720703125,812.95068359375,812.65576171875,812.545166015625,812.507568359375,812.53125,812.58935546875,812.66357421875,812.736328125],"text":["iteration:  1<br />loss: 1805.7617<br />loss_type: train_loss","iteration:  2<br />loss: 1188.3892<br />loss_type: train_loss","iteration:  3<br />loss: 1037.2322<br />loss_type: train_loss","iteration:  4<br />loss:  961.3623<br />loss_type: train_loss","iteration:  5<br />loss:  908.8958<br />loss_type: train_loss","iteration:  6<br />loss:  865.4375<br />loss_type: train_loss","iteration:  7<br />loss:  839.4365<br />loss_type: train_loss","iteration:  8<br />loss:  826.3987<br />loss_type: train_loss","iteration:  9<br />loss:  819.9087<br />loss_type: train_loss","iteration: 10<br />loss:  816.4333<br />loss_type: train_loss","iteration: 11<br />loss:  814.5986<br />loss_type: train_loss","iteration: 12<br />loss:  813.5637<br />loss_type: train_loss","iteration: 13<br />loss:  812.9507<br />loss_type: train_loss","iteration: 14<br />loss:  812.6558<br />loss_type: train_loss","iteration: 15<br />loss:  812.5452<br />loss_type: train_loss","iteration: 16<br />loss:  812.5076<br />loss_type: train_loss","iteration: 17<br />loss:  812.5312<br />loss_type: train_loss","iteration: 18<br />loss:  812.5894<br />loss_type: train_loss","iteration: 19<br />loss:  812.6636<br />loss_type: train_loss","iteration: 20<br />loss:  812.7363<br />loss_type: train_loss"],"type":"scatter","mode":"lines","line":{"width":1.8897637795275593,"color":"rgba(31,119,180,0.3)","dash":"solid"},"hoveron":"points","name":"train_loss","legendgroup":"train_loss","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],"y":[1773.143699136026,1278.1053249796603,1011.0090906184014,958.46277157623638,907.12890502183154,866.4385728987437,839.58511237723826,826.17576990678231,819.74169171339895,816.39210952240228,814.56283097875439,813.54251529210376,812.95300618106569,812.65402449896874,812.53746182355337,812.50873909444476,812.53058910117022,812.58929882027871,812.65806870201709,812.7383892054936],"text":["iteration:  1<br />loss_smooth: 1773.1437<br />loss_type: train_loss","iteration:  2<br />loss_smooth: 1278.1053<br />loss_type: train_loss","iteration:  3<br />loss_smooth: 1011.0091<br />loss_type: train_loss","iteration:  4<br />loss_smooth:  958.4628<br />loss_type: train_loss","iteration:  5<br />loss_smooth:  907.1289<br />loss_type: train_loss","iteration:  6<br />loss_smooth:  866.4386<br />loss_type: train_loss","iteration:  7<br />loss_smooth:  839.5851<br />loss_type: train_loss","iteration:  8<br />loss_smooth:  826.1758<br />loss_type: train_loss","iteration:  9<br />loss_smooth:  819.7417<br />loss_type: train_loss","iteration: 10<br />loss_smooth:  816.3921<br />loss_type: train_loss","iteration: 11<br />loss_smooth:  814.5628<br />loss_type: train_loss","iteration: 12<br />loss_smooth:  813.5425<br />loss_type: train_loss","iteration: 13<br />loss_smooth:  812.9530<br />loss_type: train_loss","iteration: 14<br />loss_smooth:  812.6540<br />loss_type: train_loss","iteration: 15<br />loss_smooth:  812.5375<br />loss_type: train_loss","iteration: 16<br />loss_smooth:  812.5087<br />loss_type: train_loss","iteration: 17<br />loss_smooth:  812.5306<br />loss_type: train_loss","iteration: 18<br />loss_smooth:  812.5893<br />loss_type: train_loss","iteration: 19<br />loss_smooth:  812.6581<br />loss_type: train_loss","iteration: 20<br />loss_smooth:  812.7384<br />loss_type: train_loss"],"type":"scatter","mode":"lines","line":{"width":3.7795275590551185,"color":"rgba(31,119,180,1)","dash":"solid"},"hoveron":"points","name":"train_loss","legendgroup":"train_loss","showlegend":false,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null}],"layout":{"margin":{"t":41.90286425902864,"r":7.3059360730593621,"b":37.260273972602747,"l":48.949771689497723},"plot_bgcolor":"rgba(255,255,255,1)","paper_bgcolor":"rgba(255,255,255,1)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724},"title":{"text":"<b> NMF Training History <\/b>","font":{"color":"rgba(0,0,0,1)","family":"","size":18.596928185969279},"x":0,"xref":"paper"},"xaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[0.049999999999999933,20.949999999999999],"tickmode":"array","ticktext":["5","10","15","20"],"tickvals":[5,10,15,20],"categoryorder":"array","categoryarray":["5","10","15","20"],"nticks":null,"ticks":"outside","tickcolor":"rgba(0,0,0,1)","ticklen":3.6529680365296811,"tickwidth":0,"showticklabels":true,"tickfont":{"color":"rgba(0,0,0,1)","family":"","size":11.68949771689498},"tickangle":-0,"showline":true,"linecolor":"rgba(0,0,0,1)","linewidth":0,"showgrid":false,"gridcolor":null,"gridwidth":0,"zeroline":false,"anchor":"y","title":{"text":"Iteration","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724}},"hoverformat":".2f"},"yaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[762.8448608398437,1855.4244262695313],"tickmode":"array","ticktext":["1000","1250","1500","1750"],"tickvals":[1000,1250,1500,1750],"categoryorder":"array","categoryarray":["1000","1250","1500","1750"],"nticks":null,"ticks":"outside","tickcolor":"rgba(0,0,0,1)","ticklen":3.6529680365296811,"tickwidth":0,"showticklabels":true,"tickfont":{"color":"rgba(0,0,0,1)","family":"","size":11.68949771689498},"tickangle":-0,"showline":true,"linecolor":"rgba(0,0,0,1)","linewidth":0,"showgrid":false,"gridcolor":null,"gridwidth":0,"zeroline":false,"anchor":"x","title":{"text":"Loss","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724}},"hoverformat":".2f"},"shapes":[{"type":"rect","fillcolor":null,"line":{"color":null,"width":0,"linetype":[]},"yref":"paper","xref":"paper","layer":"below","x0":0,"x1":1,"y0":0,"y1":1}],"showlegend":true,"legend":{"bgcolor":"rgba(255,255,255,1)","bordercolor":"transparent","borderwidth":0,"font":{"color":"rgba(0,0,0,1)","family":"","size":11.68949771689498},"title":{"text":"Loss Type","font":{"color":"rgba(0,0,0,1)","family":"","size":14.611872146118724}}},"hovermode":"closest","barmode":"relative"},"config":{"doubleClick":"reset","modeBarButtonsToAdd":["hoverclosest","hovercompare"],"showSendToCloud":false},"source":"A","attrs":{"351da02e5f2f4f":{"x":{},"y":{},"colour":{},"type":"scatter"},"351da0455e5ec6":{"x":{},"y":{},"colour":{}}},"cur_data":"351da02e5f2f4f","visdat":{"351da02e5f2f4f":["function (y) ","x"],"351da0455e5ec6":["function (y) ","x"]},"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.20000000000000001,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot","plotly_sunburstclick"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}
# Compare multiple runs
models <- replicate(5, nmf(hawaiibirds, k = 10, track_loss_history = TRUE), simplify = FALSE)
plot(models[[1]], type = "sparsity")

# }
```
