<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Facades;

use AaronFrancis\Solo\Manager;
use Illuminate\Support\Facades\Facade;

class Solo extends Facade
{
    protected static function getFacadeAccessor(): string
    {
        return Manager::class;
    }
}
