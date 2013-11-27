nnoremap <buffer> <leader>rr :wall \| SlimeSend1 %reset -f<CR>
nnoremap <buffer> <leader>r :wall \| SlimeSend1 %run -i test.py<CR>

let g:slime_default_config = {"socket_name": "default", "target_pane": "2.0"}

