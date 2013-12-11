nnoremap <buffer> <leader>r :wall \| SlimeSend1 %run -i test_spsa_rprop.py<CR>
nnoremap <leader>m  :wall\|make<CR><CR><CR>
let g:slime_default_config = {"socket_name": "default", "target_pane": "2.0"}

set efm=%C\ %.%#,%A\ \ File\ \"%f\"\\,\ line\ %l%.%#,%Z%[%^\ ]%\\@=%m
set errorformat=
    \%A\ \ File\ \"%f\"\\\,\ line\ %l\\\,%m,
    \%C\ \ \ \ %.%#,
    \%+Z%.%#Error\:\ %.%#,
    \%A\ \ File\ \"%f\"\\\,\ line\ %l,
    \%+C\ \ %.%#,
    \%-C%p^,
    \%Z%m,
    \%-G%.%#
