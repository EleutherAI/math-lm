# Proofpilev2

## The Stack
The stack is processed in `./process_source_code.py`
**Problems with the stack**
- Issue: Matlab is wrong. There are only 111 matlab files that match the regex `[a-df-zA-Z]`. Looks like most of the matlab files are just arrays saved as text. Very little of the actual code was captured. 
    - [x] Fix 1: Regex filter to delete arrays
    - [ ] Fix 2: Find rest of matlab files
- Issue: The R data contains MacOS "resource fork" files that aren't related to R at all. 
    - [x] Fix: filter out resource forks
- Issue: .sagews files have a bunch of hashes all over the place.
    - [x] Fix: sage is not worth the effort, simply delete.
- Issue: .sage files tend to have a bunch of long strings of hardcode numbers. Is this ok? e.g `ClathomasPrime/CompetitiveStableMatching:Plotting/plots.sage`
    - [x] Fix: not worth the effort, simply delete
- Issue: Wolfram mathematica has three file formats:`.wls`: Wolfram language script, handled ok; `.m`Wolfram language package, handled ok; `.nb`: notebook, the plaintext has a bunch of noise. Need to export as `.wls`. 
    - [ ] Fix: convert notebooks to tex or wls
- Issue: There is one mathematica repo, `dendaxD/QAOA-MaxCut-amplitudes`, that contains about half of all mathematica files in the stack. All these files are extremely similar and should be excluded on data diversity grounds
    - [x] Fix: repo manually deleted in `mathematica_filter()`. 
- Issue: Some maple files are actually xml
    - [x] Fix: `maple_filte()` removes xml. 
- Issue: Lots of auto-generated tex files in directories called `latex`.
    - [x] Fix: removed in `tex_filter_rexp()`
- Issue: The Julia dataset contains JSON Lines files with extension `.jl` and  large files of auto-generated boilerplate (e.g. wrappers, dumps of large arrays...).
    - [x] Fix: `julia_filter` reduces the dataset size from 1.75GB to 1.5GB.
    - [x] Fix: `julia_filter_strict` further reduces it to 1GB by using an explicit whitelist of keywords that tests for direct relevance to scientific computing.
- Issue: Jupyter notebook files contain extra JSON (e.g. with cell IDs, execution count)
    - [x] Fix: convert to Markdown. In particular, this results in a clean interleaving of text, math, and code.
- Issue: Jupyter notebook files are often too long to be tokenized.
    - Sub-issue/source: files contain long outputs; for instance, these can include dumped tables, variable values, and PNG strings. 
    - [X] Fix: remove outputs from the notebook that are base64 png strings or exception tracebacks. 

**Languages the stack does ok**:
- Lean is fine
- Python is clean (maybe get rid of Chinese characters?)

**Open questions**:
- I'm not sure if my C/C++ filtering is good at all. Am I getting too many `.h` files?
- Do we want Chinese comments in our Python?
- Currently the jupyter notebook filtering is strict. We can add more conditions if we want more data.
    
