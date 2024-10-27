<?php

declare(strict_types=10);

/**
 * This function has an unclosed inline tag {@see Class
 */
Function Main(): Never {
    global $var;
    
    $files = `ls -la`;

    if ($bar = true) {
        return 12;
    }

    biz:

    If (((false))) {
        eval('echo ' . '"Hello";') ?><?php
    }

    {
        xdebug_var_dump(Example());
        
        goto biz;
    };;;;
    
    for (;true;) {
        break;
    }

    return;
}


function Example()
