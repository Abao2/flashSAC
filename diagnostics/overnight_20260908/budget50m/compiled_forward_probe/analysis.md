# Actual CUDA/Inductor forward parity

Measured differences; no arbitrary exact-zero pass criterion. No physics or optimizer updates.

| phase | batch | stride | mu max/RMSE | sigma max/RMSE | tanh mean max | fixed-noise action max |
| --- | ---: | --- | --- | --- | ---: | ---: |
| reset | 32 | strided324 | 0.000607252/5.24956e-05 | 0.00160384/0.000460408 | 0.000376433 | 0.00144252 |
| reset | 32 | contiguous162 | 0.000607252/5.24956e-05 | 0.00160384/0.000460408 | 0.000376433 | 0.00144252 |
| reset | 1024 | strided324 | 0.000287294/7.00586e-06 | 0.00160116/0.000549801 | 0.000106573 | 0.00282913 |
| reset | 1024 | contiguous162 | 0.000287294/7.00586e-06 | 0.00160116/0.000549801 | 0.000106573 | 0.00282913 |
| lifted_proxy | 32 | strided324 | 0.00140095/0.000132178 | 0.00137836/0.000349101 | 0.000891805 | 0.00270535 |
| lifted_proxy | 32 | contiguous162 | 0.00140095/0.000132178 | 0.00137836/0.000349101 | 0.000891805 | 0.00270535 |
| lifted_proxy | 1024 | strided324 | 0.00266105/6.57674e-05 | 0.00360751/0.000365944 | 0.00256611 | 0.00346735 |
| lifted_proxy | 1024 | contiguous162 | 0.00266105/6.57674e-05 | 0.00360751/0.000365944 | 0.00256611 | 0.00346735 |
| near_goal_center_proxy | 32 | strided324 | 0.000851989/9.87992e-05 | 0.00192738/0.000317219 | 0.000420541 | 0.002156 |
| near_goal_center_proxy | 32 | contiguous162 | 0.000851989/9.87992e-05 | 0.00192738/0.000317219 | 0.000420541 | 0.002156 |
| near_goal_center_proxy | 1024 | strided324 | 0.0016439/6.7981e-05 | 0.00193721/0.000307051 | 0.00158243 | 0.00320524 |
| near_goal_center_proxy | 1024 | contiguous162 | 0.0016439/6.7981e-05 | 0.00193721/0.000307051 | 0.00158243 | 0.00320524 |

24 sampler comparisons (12 calls ×2 batches), same RNG/noise caches. Max action difference: 0.00512135.
Actor/native critic/target/temp parameters and BN buffers unchanged and finite. JSON retains actual input strides, compiler counters, all source hashes and descriptive tolerances.
