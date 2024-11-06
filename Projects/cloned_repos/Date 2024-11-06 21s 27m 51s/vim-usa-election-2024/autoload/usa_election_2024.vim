let s:status = ''
let s:res = []

function! usa_election_2024#status()
  return s:status
endfunction

function! s:job_callback(ch, data) abort
  call add(s:res, a:data)
  if ch_status(a:ch) !=# 'closed'
    return
  endif
  try
    let l:res = json_decode(join(s:res, '')).candidates
    let l:harris = filter(copy(l:res), {_, x->x['last_name'] == 'Harris'})[0].electoral_votes_total
    let l:trump = filter(copy(l:res), {_, x->x['last_name'] == 'Trump'})[0].electoral_votes_total
    let s:status = printf('Harris(%d) vs Trump(%d)', l:harris, l:trump)
  catch
    echomsg v:exception
  finally
    let s:res = []
  endtry
  call timer_start(60000, {x->usa_election_2024#update()})
endfunction

function! usa_election_2024#update()
  call job_start(['curl', '-s', 'https://data.ddhq.io/electoral_college/2024'], {'callback': function('s:job_callback')})
endfunction

call usa_election_2024#update()
