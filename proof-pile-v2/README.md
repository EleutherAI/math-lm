# Proofpilev2

## The Stack
The stack is processed in `./thestack.py`
**Problems with the stack**
- Issue: Matlab is wrong. There are only 111 matlab files that match the regex `[a-df-zA-Z]`. Looks like most of the matlab files are just arrays saved as text. Very little of the actual code was captured. 
    - [x] Fix 1: Regex filter to delete arrays
    - [ ] Fix 2: Find rest of matlab files
- Issue: The R data contains MacOS "resource fork" files that aren't related to R at all. 
    - [x] Fix: filter out resource forks
- Issue: .sagews files have a bunch of hashes all over the place.
    - [ ] Fix: figure out how to delete hashes, or render notebooks. 
- Issue: .sage files tend to have a bunch of long strings of hardcode numbers. Is this ok? e.g `ClathomasPrime/CompetitiveStableMatching:Plotting/plots.sage`
- Issue: Wolfram mathematica has three file formats:`.wls`: Wolfram language script, handled ok; `.m`Wolfram language package, handled ok; `.nb`: notebook, the plaintext has a bunch of noise. Need to export as `.wls`. 
    - [ ] Fix: convert notebooks to tex or wls
- Issue: There is one mathematica repo, `dendaxD/QAOA-MaxCut-amplitudes`, that contains about half of all mathematica files in the stack. All these files are extremely similar and should be included on data diversity grounds
    - [x] Fix: repo manually deleted in `mathematica_filter()`. 
- Issue: Some maple files are actually xml
    - [x] Fix: `maple_filte()` removes xml. 
- Issue: Lots of auto-generated tex files in directories called `latex`.
    - [x] Fix: removed in `tex_filter_rexp()`

**Languages the stack does ok**:
- Lean is fine
- Julia is fine (possibly want to remove files that meet jsonl spec)
- Python is clean (maybe get rid of Chinese characters?)

**Open questions**:
- I'm not sure if my C/C++ filtering is good at all. Am I getting too many `.h` files?
- Do we want Chinese comments in our Python?
    
