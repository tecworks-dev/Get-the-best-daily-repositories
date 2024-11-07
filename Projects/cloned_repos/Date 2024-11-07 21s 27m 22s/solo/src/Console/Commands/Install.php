<?php
/**
 * @author Aaron Francis <aarondfrancis@gmail.com>
 * @link https://aaronfrancis.com
 * @link https://twitter.com/aarondfrancis
 */

namespace AaronFrancis\Solo\Console\Commands;

use Illuminate\Console\Command;
use Illuminate\Support\ServiceProvider;
use Illuminate\Support\Str;

class Install extends Command
{
    protected $signature = 'solo:install';

    protected $description = 'Install the Solo service provider';

    public function handle()
    {
        $this->comment('Publishing Solo Service Provider...');
        $this->callSilent('vendor:publish', ['--tag' => 'solo-provider']);

        $this->registerSoloServiceProvider();

        $this->info('Solo installed successfully.');
        $this->info('Run `php artisan solo` to start.');
    }

    /**
     * Register the Solo service provider in the application configuration file.
     *
     * @return void
     */
    protected function registerSoloServiceProvider()
    {
        $namespace = Str::replaceLast('\\', '', $this->laravel->getNamespace());

        if (file_exists($this->laravel->bootstrapPath('providers.php'))) {
            ServiceProvider::addProviderToBootstrapFile("{$namespace}\\Providers\\SoloServiceProvider");
        } else {
            $appConfig = file_get_contents(config_path('app.php'));

            if (Str::contains($appConfig, $namespace . '\\Providers\\SoloServiceProvider::class')) {
                return;
            }

            file_put_contents(config_path('app.php'), str_replace(
                "{$namespace}\\Providers\EventServiceProvider::class," . PHP_EOL,
                "{$namespace}\\Providers\EventServiceProvider::class," . PHP_EOL . "        {$namespace}\Providers\SoloServiceProvider::class," . PHP_EOL,
                $appConfig
            ));
        }

        file_put_contents(app_path('Providers/SoloServiceProvider.php'), str_replace(
            "namespace App\Providers;",
            "namespace {$namespace}\Providers;",
            file_get_contents(app_path('Providers/SoloServiceProvider.php'))
        ));
    }
}
