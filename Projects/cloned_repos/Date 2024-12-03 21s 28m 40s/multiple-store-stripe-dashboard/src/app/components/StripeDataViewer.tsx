'use client';

import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import {
  Area,
  AreaChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Ban, EyeOff, Eye, Loader2, Search, DollarSign, CreditCard, Users, TrendingUp, Trophy, CalendarClock, Key, Save } from "lucide-react";
import { format, parseISO } from 'date-fns'
import { toZonedTime } from 'date-fns-tz'
import { Switch } from "@/components/ui/switch"
import { Label as UILabel } from "@/components/ui/label"
import { ThemeToggle } from '../components/theme-toggle';
import { Input } from "@/components/ui/input";

interface PaymentData {
  id: string;
  amount: number;
  currency: string;
  created: number;
  paymentType: 'one_time' | 'subscription';
  subscriptionInterval?: 'month' | 'year';
  subscriptionType?: 'new' | 'renewal';
  status: string;
}

interface MerchantData {
  merchantId: string;
  totalOrders: number;
  totalAmount: number;
  currency: string;
  payments: PaymentData[];
  accountName: string;
  upcomingRenewals: UpcomingRenewal[];
}

interface UpcomingRenewal {
  customerId: string;
  amount: number;
  currency: string;
  renewalDate: number;
}

type MetricType = 'orders' | 'revenue' | 'new_subs_count' | 'renewal_count' | 'new_subs_amount' | 'renewal_amount';

type ViewMode = 'all' | 'new_subscriptions' | 'renewals';

const getBrowserTimezone = () => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone;
  } catch (e) {
    return 'UTC';
  }
};

const isDateInSelectedMonth = (timestamp: number, year: string, month: string, timezone: string): boolean => {
  const date = toZonedTime(new Date(timestamp * 1000), timezone);
  const monthNum = parseInt(month);
  return date.getMonth() + 1 === monthNum && date.getFullYear() === parseInt(year);
};

const getDaysInMonth = (year: number, month: number): string[] => {
  const daysInMonth = new Date(year, month, 0).getDate();
  return Array.from(
    { length: daysInMonth },
    (_, i) => format(new Date(year, month - 1, i + 1), 'yyyy-MM-dd')
  );
};

export default function StripeDataViewer() {
  const now = new Date();
  const [year, setYear] = useState(now.getFullYear().toString());
  const [month, setMonth] = useState((now.getMonth() + 1).toString().padStart(2, '0'));
  const [day, setDay] = useState('');
  const [timezone, setTimezone] = useState('UTC');
  const [merchantsData, setMerchantsData] = useState<MerchantData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('all');
  const [showSensitiveData, setShowSensitiveData] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState<{ current: number; total: number } | null>(null);
  const [showOrders, setShowOrders] = useState(true);
  const [showRevenue, setShowRevenue] = useState(true);
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isPasswordSaved, setIsPasswordSaved] = useState(false);

  useEffect(() => {
    const savedTimezone = localStorage.getItem('preferredTimezone');
    setTimezone(savedTimezone || getBrowserTimezone());
    
    const savedShowSensitive = localStorage.getItem('showSensitiveData');
    setShowSensitiveData(savedShowSensitive === 'true');
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('preferredTimezone', timezone);
    }
  }, [timezone]);

  useEffect(() => {
    const savedPassword = localStorage.getItem('stripeViewerPassword');
    if (savedPassword) {
      setPassword(savedPassword);
      setIsPasswordSaved(true);
    }
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('showSensitiveData', showSensitiveData.toString());
    }
  }, [showSensitiveData]);

  const fetchData = async (targetDate: string) => {
    try {
      setLoading(true);
      setError(null);
      setMerchantsData([]);
      setLoadingProgress(null);

      const isMockMode = process.env.NEXT_PUBLIC_USE_MOCK_DATA === 'true';
      
      if (isMockMode) {
        try {
          const response = await fetch('/api/mock');
          const mockData = await response.json();
          
          setLoadingProgress({ current: 0, total: mockData.length });
          
          for (let i = 0; i < mockData.length; i++) {
            await new Promise(resolve => setTimeout(resolve, 500));
            setLoadingProgress({ current: i + 1, total: mockData.length });
            setMerchantsData(prev => [...prev, mockData[i]]);
          }
          
          setLoading(false);
          setError(null);
        } catch (error) {
          console.error('Error fetching mock data:', error);
          setError('Failed to fetch mock data');
          setLoading(false);
        }
      } else {
        const eventSource = new EventSource(
          `/api/stripe-data?date=${targetDate}&timezone=${timezone}&password=${encodeURIComponent(password)}`
        );

        eventSource.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            
            if (data.error) {
              eventSource.close();
              setError(data.error);
              setLoading(false);
              return;
            }
            
            if (data.type === 'total') {
              setLoadingProgress({ current: 0, total: data.count });
            } else if (data.type === 'data') {
              setLoadingProgress({ current: data.current, total: data.total });
              setMerchantsData(prev => [...prev, data.data]);
              
              if (data.current === data.total) {
                eventSource.close();
                setLoading(false);
                setError(null);
              }
            }
          } catch (e) {
            console.error('Error parsing merchant data:', e);
            eventSource.close();
            setError('Failed to parse data');
            setLoading(false);
          }
        };

        eventSource.onerror = (error) => {
          console.error('EventSource error:', error);
          eventSource.close();
          setError('Failed to fetch data. Please check your password and try again.');
          setLoading(false);
        };

        return () => {
          eventSource.close();
        };
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number, currency: string) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency.toUpperCase(),
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount / 100);
  };


  const handleDateSubmit = () => {
    const targetDate = day ? `${year}-${month}-${day}` : `${year}-${month}`;
    fetchData(targetDate);
  };

  const formatDateWithTimezone = (dateStr: string) => {
    const zonedDate = toZonedTime(parseISO(dateStr), timezone);
    return format(zonedDate, 'yyyy-MM-dd');
  };

  const prepareChartData = (merchant: MerchantData) => {
    const dailyData: { [key: string]: any } = {};
    
    merchant.payments.forEach(payment => {
      const zonedDate = toZonedTime(new Date(payment.created * 1000), timezone);
      const dateStr = format(zonedDate, 'yyyy-MM-dd');
      
      if (!dailyData[dateStr]) {
        dailyData[dateStr] = {
          date: dateStr,
          orders: 0,
          revenue: 0,
          new_subs_count: 0,
          renewal_count: 0,
          new_subs_amount: 0,
          renewal_amount: 0,
        };
      }
      
      dailyData[dateStr].orders += 1;
      dailyData[dateStr].revenue += payment.amount / 100;

      if (payment.paymentType === 'one_time') {
        dailyData[dateStr].one_time_count += 1;
        dailyData[dateStr].one_time_amount += payment.amount / 100;
      } else if (payment.paymentType === 'subscription') {
        if (payment.subscriptionType === 'new') {
          dailyData[dateStr].new_subs_count += 1;
          dailyData[dateStr].new_subs_amount += payment.amount / 100;
        } else {
          dailyData[dateStr].renewal_count += 1;
          dailyData[dateStr].renewal_amount += payment.amount / 100;
        }
      }
    });

    return Object.values(dailyData).sort((a, b) => a.date.localeCompare(b.date));
  };

  const getMetricLabel = (metric: MetricType): string => {
    const labels: Record<MetricType, string> = {
      orders: 'Order Count',
      revenue: 'Total Revenue',
      new_subs_count: 'New Subscriptions',
      renewal_count: 'Renewal Count',
      new_subs_amount: 'New Subscription Revenue',
      renewal_amount: 'Renewal Revenue'
    };
    return labels[metric];
  };

  const hasMerchantRevenue = (merchant: MerchantData) => {
    return merchant.payments.length > 0 && merchant.totalAmount > 0;
  };

  const preparePieChartData = (merchant: MerchantData, type: 'amount' | 'count') => {
    const totals = {
      one_time: { value: 0, label: 'One-time Payments' },
      new_subs: { value: 0, label: 'New Subscriptions' },
      renewals: { value: 0, label: 'Renewals' }
    };

    merchant.payments.forEach(payment => {
      if (viewMode === 'new_subscriptions' && 
          !(payment.paymentType === 'subscription' && payment.subscriptionType === 'new')) {
        return;
      }
      if (viewMode === 'renewals' && 
          !(payment.paymentType === 'subscription' && payment.subscriptionType === 'renewal')) {
        return;
      }

      if (payment.paymentType === 'one_time') {
        totals.one_time.value += type === 'amount' ? payment.amount : 1;
      } else if (payment.paymentType === 'subscription') {
        if (payment.subscriptionType === 'new') {
          totals.new_subs.value += type === 'amount' ? payment.amount : 1;
        } else {
          totals.renewals.value += type === 'amount' ? payment.amount : 1;
        }
      }
    });

    return Object.values(totals)
      .filter(item => item.value > 0)
      .map(item => ({
        ...item,
        value: type === 'amount' ? item.value / 100 : item.value
      }));
  };

  const prepareAggregatedChartData = () => {
    const allDays = getDaysInMonth(parseInt(year), parseInt(month));
    const dailyData: { [key: string]: any } = {};
    
    allDays.forEach(dateStr => {
      dailyData[dateStr] = {
        date: dateStr,
        orders: 0,
        revenue: 0,
        new_subs_count: 0,
        renewal_count: 0,
        new_subs_amount: 0,
        renewal_amount: 0,
      };
    });
    
    merchantsData.forEach(merchant => {
      merchant.payments.forEach(payment => {
        const zonedDate = toZonedTime(new Date(payment.created * 1000), timezone);
        
        if (isDateInSelectedMonth(payment.created, year, month, timezone)) {
          const dateStr = format(zonedDate, 'yyyy-MM-dd');
          
          dailyData[dateStr].orders += 1;
          dailyData[dateStr].revenue += payment.amount / 100;

          if (payment.paymentType === 'subscription') {
            if (payment.subscriptionType === 'new') {
              dailyData[dateStr].new_subs_count += 1;
              dailyData[dateStr].new_subs_amount += payment.amount / 100;
            } else {
              dailyData[dateStr].renewal_count += 1;
              dailyData[dateStr].renewal_amount += payment.amount / 100;
            }
          }
        }
      });
    });

    return Object.values(dailyData).sort((a, b) => a.date.localeCompare(b.date));
  };


  const maskSensitiveData = (value: number | string, type: 'amount' | 'name') => {
    if (!showSensitiveData) {
      if (type === 'amount') return '***';
      if (type === 'name') return `Store ${value}`;
    }
    return value;
  };

  const isSpecificDayView = () => {
    return !!day && day !== 'all';
  };

  const totalRevenue = useMemo(() => 
    merchantsData.reduce((sum, merchant) => 
      sum + merchant.payments.reduce((acc, payment) => acc + payment.amount, 0), 0
    ) / 100, [merchantsData]);

  const totalOrders = useMemo(() => 
    merchantsData.reduce((sum, merchant) => 
      sum + merchant.payments.length, 0
    ), [merchantsData]);

  const handleSavePassword = () => {
    localStorage.setItem('stripeViewerPassword', password);
    setIsPasswordSaved(true);
  };

  return (
    <div className="p-2 max-w-[1920px] mx-auto">
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <CardTitle className="text-2xl font-bold">Multiple-Store Stripe Insights</CardTitle>
            <div className="flex items-center gap-3 flex-wrap sm:flex-1 sm:justify-end">
              <div className="relative flex items-center">
                <Key className="absolute left-2 h-4 w-4 text-muted-foreground" />
                <Input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    setIsPasswordSaved(false);
                  }}
                  className="w-[240px] pl-8 pr-10"
                  placeholder="Enter password"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-8 top-0 h-full px-3"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3"
                  onClick={handleSavePassword}
                  disabled={isPasswordSaved}
                >
                  <Save className={`h-4 w-4 ${isPasswordSaved ? 'text-muted-foreground' : 'text-primary'}`} />
                </Button>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSensitiveData(!showSensitiveData)}
                className="flex items-center gap-2"
              >
                {showSensitiveData ? (
                  <>
                    <EyeOff className="h-4 w-4" />
                    <span>Hide Sensitive Data</span>
                  </>
                ) : (
                  <>
                    <Eye className="h-4 w-4" />
                    <span>Show Sensitive Data</span>
                  </>
                )}
              </Button>
              <ThemeToggle />
            </div>
          </div>
        </CardHeader>
        <CardContent className="pb-2">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
              <div className="flex-1">
                <div className="flex flex-col gap-2">
                  <span className="text-sm text-muted-foreground">Date Range:</span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                    <Select value={year} onValueChange={setYear}>
                      <SelectTrigger>
                        <SelectValue placeholder="Year" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 5 }, (_, i) => now.getFullYear() - i).map(y => (
                          <SelectItem key={y} value={y.toString()}>{y}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Select value={month} onValueChange={setMonth}>
                      <SelectTrigger>
                        <SelectValue placeholder="Month" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 12 }, (_, i) => (i + 1).toString().padStart(2, '0')).map(m => (
                          <SelectItem key={m} value={m}>{m}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Select value={day} onValueChange={setDay}>
                      <SelectTrigger>
                        <SelectValue placeholder="All Days" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Days</SelectItem>
                        {Array.from(
                          { length: new Date(parseInt(year), parseInt(month), 0).getDate() },
                          (_, i) => (i + 1).toString().padStart(2, '0')
                        ).map(d => (
                          <SelectItem key={d} value={d}>{d}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Select value={timezone} onValueChange={setTimezone}>
                      <SelectTrigger>
                        <SelectValue placeholder="Timezone" />
                      </SelectTrigger>
                      <SelectContent>
                        {timezone !== getBrowserTimezone() && (
                          <SelectItem value={getBrowserTimezone()}>Browser Default ({getBrowserTimezone()})</SelectItem>
                        )}
                        <SelectItem value="UTC">UTC</SelectItem>
                        <SelectItem value="America/New_York">America/New_York</SelectItem>
                        <SelectItem value="America/Los_Angeles">America/Los_Angeles</SelectItem>
                        <SelectItem value="Asia/Shanghai">Asia/Shanghai</SelectItem>
                        <SelectItem value="Asia/Tokyo">Asia/Tokyo</SelectItem>
                        <SelectItem value="Europe/London">Europe/London</SelectItem>
                        <SelectItem value="Europe/Paris">Europe/Paris</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <Button 
                onClick={handleDateSubmit} 
                disabled={loading}
                variant="default"
                size="lg"
                className="w-full sm:w-[140px] bg-primary hover:bg-primary/90"
              >
                {loading ? (
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {loadingProgress 
                      ? `${loadingProgress.current}/${loadingProgress.total}`
                      : "Loading"
                    }
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    <span>Fetch Data</span>
                  </div>
                )}
              </Button>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <div className="flex-1">
                <div className="flex flex-col gap-2">
                  <span className="text-sm text-muted-foreground">Filter Type:</span>
                  <div className="grid grid-cols-3 gap-2">
                    <Button 
                      size="sm"
                      variant={viewMode === 'all' ? 'secondary' : 'outline'}
                      onClick={() => setViewMode('all')}
                      className="w-full"
                    >
                      All Orders
                    </Button>
                    <Button 
                      size="sm"
                      variant={viewMode === 'new_subscriptions' ? 'secondary' : 'outline'}
                      onClick={() => setViewMode('new_subscriptions')}
                      className="w-full"
                    >
                      New Subs
                    </Button>
                    <Button 
                      size="sm"
                      variant={viewMode === 'renewals' ? 'secondary' : 'outline'}
                      onClick={() => setViewMode('renewals')}
                      className="w-full"
                    >
                      Renewals
                    </Button>
                  </div>
                </div>
              </div>

              <div className="min-w-[200px]">
                <div className="flex flex-col gap-2">
                  <span className="text-sm text-muted-foreground">Show Data:</span>
                  <div className="flex items-center gap-4 h-9">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="show-orders"
                        checked={showOrders}
                        onCheckedChange={(checked) => {
                          if (!checked && !showRevenue) {
                            setShowRevenue(true);
                          }
                          setShowOrders(checked);
                        }}
                      />
                      <UILabel htmlFor="show-orders">Orders</UILabel>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Switch
                        id="show-revenue"
                        checked={showRevenue}
                        onCheckedChange={(checked) => {
                          if (!checked && !showOrders) {
                            setShowOrders(true);
                          }
                          setShowRevenue(checked);
                        }}
                      />
                      <UILabel htmlFor="show-revenue">Revenue</UILabel>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="mb-4 border-red-500">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-red-500">
              <Ban className="h-4 w-4" />
              <span>{error}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {merchantsData.length > 0 && (
        <Card className="mb-4">
          <CardHeader className="pb-2">
            <CardTitle>Overall Performance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="bg-blue-100 dark:bg-blue-900/50 p-2 rounded-lg">
                      <DollarSign className="h-4 w-4 text-blue-500" />
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-sm text-muted-foreground">Revenue & Orders</p>
                      <p className="text-xl font-bold text-blue-500">
                        {showSensitiveData 
                          ? formatCurrency(totalRevenue * 100, merchantsData[0]?.currency || 'usd')
                          : '***'}
                      </p>
                      {showSensitiveData && (
                        <p className="text-xs text-muted-foreground">
                          Orders: {totalOrders.toLocaleString()}
                        </p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="bg-green-100 dark:bg-green-900/50 p-2 rounded-lg">
                      <Users className="h-4 w-4 text-green-500" />
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-sm text-muted-foreground">Active Merchants</p>
                      <p className="text-xl font-bold text-green-500">
                        {showSensitiveData ? merchantsData.length.toString() : '***'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="bg-purple-100 dark:bg-purple-900/50 p-2 rounded-lg">
                      <TrendingUp className="h-4 w-4 text-purple-500" />
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-sm text-muted-foreground">Average Order Value</p>
                      <p className="text-xl font-bold text-purple-500">
                        {showSensitiveData 
                          ? formatCurrency((totalRevenue / totalOrders) * 100, merchantsData[0]?.currency || 'usd')
                          : '***'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card/50">
                <CardContent className="p-4">
                  <div className="flex items-center gap-3">
                    <div className="bg-orange-100 dark:bg-orange-900/50 p-2 rounded-lg">
                      <CalendarClock className="h-4 w-4 text-orange-500" />
                    </div>
                    <div className="space-y-0.5">
                      <p className="text-sm text-muted-foreground">Upcoming Renewals</p>
                      <p className="text-xl font-bold text-orange-500">
                        {showSensitiveData 
                          ? formatCurrency(
                              merchantsData.reduce((total, merchant) => 
                                total + merchant.upcomingRenewals.reduce((sum, renewal) => 
                                  sum + renewal.amount, 0
                                ), 0
                              ),
                              merchantsData[0]?.currency || 'usd'
                            )
                          : '***'}
                      </p>
                      {showSensitiveData && (
                        <p className="text-xs text-muted-foreground">
                          Count: {merchantsData.reduce((total, merchant) => 
                            total + merchant.upcomingRenewals.length, 0
                          ).toLocaleString()}
                        </p>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="flex flex-col gap-4 mt-4">
              {!isSpecificDayView() && (
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={prepareAggregatedChartData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="date" 
                        tickFormatter={(dateStr) => {
                          const [year, month, day] = dateStr.split('-');
                          return `${month}/${day}`;
                        }}
                      />
                      <YAxis 
                        yAxisId="count"
                        orientation="left"
                        tickFormatter={(value) => 
                          showSensitiveData 
                            ? value.toLocaleString()
                            : '***'
                        }
                      />
                      <YAxis 
                        yAxisId="amount"
                        orientation="right"
                        tickFormatter={(value) => 
                          showSensitiveData
                            ? new Intl.NumberFormat('en-US', {
                                style: 'currency',
                                currency: 'USD',
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0,
                              }).format(value).replace('$', '')
                            : '***'
                        }
                      />
                      <Tooltip
                        labelFormatter={(dateStr) => formatDateWithTimezone(dateStr)}
                        formatter={(value: number, name: any) => {
                          if (String(name).includes('amount') || String(name) === 'revenue') {
                            return [
                              showSensitiveData 
                                ? formatCurrency(value * 100, merchantsData[0]?.currency || 'usd')
                                : '***',
                              name
                            ];
                          }
                          return [
                            showSensitiveData 
                              ? value.toLocaleString()
                              : '***',
                            name
                          ];
                        }}
                      />
                      <Legend 
                        layout="horizontal"
                        align="center"
                        verticalAlign="bottom"
                        wrapperStyle={{ paddingTop: '20px' }}
                      />
                      {showOrders && (
                        <Area
                          yAxisId="count"
                          type="monotone"
                          dataKey="orders"
                          name="Orders"
                          fill="hsl(var(--chart-1))"
                          fillOpacity={0.4}
                          stroke="hsl(var(--chart-1))"
                        />
                      )}
                      {showRevenue && (
                        <Area
                          yAxisId="amount"
                          type="monotone"
                          dataKey="revenue"
                          name="Revenue"
                          fill="hsl(var(--chart-2))"
                          fillOpacity={0.4}
                          stroke="hsl(var(--chart-2))"
                        />
                      )}
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              )}

              <div className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <div className="space-x-4 flex items-center">
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-1))]" />
                        <span className="text-muted-foreground">One-time</span>
                        <span className="font-medium">
                          {showSensitiveData 
                            ? merchantsData.reduce((sum, merchant) => 
                                sum + merchant.payments.filter(p => p.paymentType === 'one_time').length, 0
                              )
                            : '***'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-2))]" />
                        <span className="text-muted-foreground">New Subs</span>
                        <span className="font-medium">
                          {showSensitiveData 
                            ? merchantsData.reduce((sum, merchant) => 
                                sum + merchant.payments.filter(p => 
                                  p.paymentType === 'subscription' && p.subscriptionType === 'new'
                                ).length, 0
                              )
                            : '***'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-3))]" />
                        <span className="text-muted-foreground">Renewals</span>
                        <span className="font-medium">
                          {showSensitiveData 
                            ? merchantsData.reduce((sum, merchant) => 
                                sum + merchant.payments.filter(p => 
                                  p.paymentType === 'subscription' && p.subscriptionType === 'renewal'
                                ).length, 0
                              )
                            : '***'}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="h-2 bg-muted rounded-full overflow-hidden flex">
                    {showSensitiveData && (
                      <>
                        <div 
                          className="h-full bg-[hsl(var(--chart-1))]" 
                          style={{ 
                            width: `${(merchantsData.reduce((sum, merchant) => 
                              sum + merchant.payments.filter(p => p.paymentType === 'one_time').length, 0
                            ) / totalOrders * 100)}%` 
                          }}
                        />
                        <div 
                          className="h-full bg-[hsl(var(--chart-2))]" 
                          style={{ 
                            width: `${(merchantsData.reduce((sum, merchant) => 
                              sum + merchant.payments.filter(p => 
                                p.paymentType === 'subscription' && p.subscriptionType === 'new'
                              ).length, 0
                            ) / totalOrders * 100)}%` 
                          }}
                        />
                        <div 
                          className="h-full bg-[hsl(var(--chart-3))]" 
                          style={{ 
                            width: `${(merchantsData.reduce((sum, merchant) => 
                              sum + merchant.payments.filter(p => 
                                p.paymentType === 'subscription' && p.subscriptionType === 'renewal'
                              ).length, 0
                            ) / totalOrders * 100)}%` 
                          }}
                        />
                      </>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center text-sm">
                    <div className="space-x-4 flex items-center">
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-1))]" />
                        <span className="text-muted-foreground">One-time</span>
                        <span className="font-medium">
                          {showSensitiveData 
                            ? formatCurrency(
                                merchantsData.reduce((sum, merchant) => 
                                  sum + merchant.payments
                                    .filter(p => p.paymentType === 'one_time')
                                    .reduce((acc, p) => acc + p.amount, 0), 0
                                ),
                                merchantsData[0]?.currency || 'usd'
                              )
                            : '***'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-2))]" />
                        <span className="text-muted-foreground">Monthly</span>
                        <span className="font-medium">
                          {showSensitiveData 
                            ? formatCurrency(
                                merchantsData.reduce((sum, merchant) => 
                                  sum + merchant.payments
                                    .filter(p => p.paymentType === 'subscription' && p.subscriptionInterval === 'month')
                                    .reduce((acc, p) => acc + p.amount, 0), 0
                                ),
                                merchantsData[0]?.currency || 'usd'
                              )
                            : '***'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-3))]" />
                        <span className="text-muted-foreground">Yearly</span>
                        <span className="font-medium">
                          {showSensitiveData 
                            ? formatCurrency(
                                merchantsData.reduce((sum, merchant) => 
                                  sum + merchant.payments
                                    .filter(p => p.paymentType === 'subscription' && p.subscriptionInterval === 'year')
                                    .reduce((acc, p) => acc + p.amount, 0), 0
                                ),
                                merchantsData[0]?.currency || 'usd'
                              )
                            : '***'}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="h-2 bg-muted rounded-full overflow-hidden flex">
                    {showSensitiveData && (
                      <>
                        <div 
                          className="h-full bg-[hsl(var(--chart-1))]" 
                          style={{ 
                            width: `${(merchantsData.reduce((sum, merchant) => 
                              sum + merchant.payments
                                .filter(p => p.paymentType === 'one_time')
                                .reduce((acc, p) => acc + p.amount, 0), 0
                            ) / (totalRevenue * 100) * 100)}%` 
                          }}
                        />
                        <div 
                          className="h-full bg-[hsl(var(--chart-2))]" 
                          style={{ 
                            width: `${(merchantsData.reduce((sum, merchant) => 
                              sum + merchant.payments
                                .filter(p => p.paymentType === 'subscription' && p.subscriptionInterval === 'month')
                                .reduce((acc, p) => acc + p.amount, 0), 0
                            ) / (totalRevenue * 100) * 100)}%` 
                          }}
                        />
                        <div 
                          className="h-full bg-[hsl(var(--chart-3))]" 
                          style={{ 
                            width: `${(merchantsData.reduce((sum, merchant) => 
                              sum + merchant.payments
                                .filter(p => p.paymentType === 'subscription' && p.subscriptionInterval === 'year')
                                .reduce((acc, p) => acc + p.amount, 0), 0
                            ) / (totalRevenue * 100) * 100)}%` 
                          }}
                        />
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        {merchantsData.map((merchant, index) => (
          <Card key={merchant.merchantId} className="mb-0">
            <CardHeader className="pb-2">
              <CardTitle>
                {maskSensitiveData(index + 1, 'name')}
                {showSensitiveData && `: ${merchant.accountName}`}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {hasMerchantRevenue(merchant) ? (
                <div className="flex flex-col gap-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Revenue & Orders</p>
                      <p className="text-xl font-bold">
                        {showSensitiveData 
                          ? formatCurrency(merchant.totalAmount, merchant.currency)
                          : '***'}
                      </p>
                      {showSensitiveData && (
                        <p className="text-xs text-muted-foreground">
                          Orders: {merchant.totalOrders.toLocaleString()}
                        </p>
                      )}
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm text-muted-foreground">Upcoming Renewals</p>
                      <p className="text-xl font-bold text-orange-500">
                        {showSensitiveData 
                          ? formatCurrency(
                              merchant.upcomingRenewals.reduce((sum, renewal) => sum + renewal.amount, 0),
                              merchant.currency
                            )
                          : '***'}
                      </p>
                      {showSensitiveData && (
                        <p className="text-xs text-muted-foreground">
                          Count: {merchant.upcomingRenewals.length}
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center text-sm">
                      <div className="space-x-4 flex items-center">
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-1))]" />
                          <span className="text-muted-foreground">One-time</span>
                          <span className="font-medium">
                            {showSensitiveData 
                              ? merchant.payments.filter(p => p.paymentType === 'one_time').length
                              : '***'}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-2))]" />
                          <span className="text-muted-foreground">New Subs</span>
                          <span className="font-medium">
                            {showSensitiveData 
                              ? merchant.payments.filter(p => p.paymentType === 'subscription' && p.subscriptionType === 'new').length
                              : '***'}
                            </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <div className="w-3 h-3 rounded-sm bg-[hsl(var(--chart-3))]" />
                          <span className="text-muted-foreground">Renewals</span>
                          <span className="font-medium">
                            {showSensitiveData 
                              ? merchant.payments.filter(p => p.paymentType === 'subscription' && p.subscriptionType === 'renewal').length
                              : '***'}
                            </span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="h-2 bg-muted rounded-full overflow-hidden flex">
                      {showSensitiveData && (
                        <>
                          <div 
                            className="h-full bg-[hsl(var(--chart-1))]" 
                            style={{ 
                              width: `${(merchant.payments.filter(p => p.paymentType === 'one_time').length / merchant.payments.length * 100)}%` 
                            }}
                          />
                          <div 
                            className="h-full bg-[hsl(var(--chart-2))]" 
                            style={{ 
                              width: `${(merchant.payments.filter(p => p.paymentType === 'subscription' && p.subscriptionType === 'new').length / merchant.payments.length * 100)}%` 
                            }}
                          />
                          <div 
                            className="h-full bg-[hsl(var(--chart-3))]" 
                            style={{ 
                              width: `${(merchant.payments.filter(p => p.paymentType === 'subscription' && p.subscriptionType === 'renewal').length / merchant.payments.length * 100)}%` 
                            }}
                          />
                        </>
                      )}
                    </div>
                  </div>

                  {!isSpecificDayView() && (
                    <div className="h-[200px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={prepareChartData(merchant)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="date" 
                            tickFormatter={(dateStr) => {
                              const [year, month, day] = dateStr.split('-');
                              return `${month}/${day}`;
                            }}
                          />
                          <YAxis 
                            yAxisId="revenue"
                            orientation="left"
                            tickFormatter={(value) => 
                              showSensitiveData 
                                ? formatCurrency(value * 100, merchant.currency).replace(merchant.currency.toUpperCase(), '')
                                : '***'
                            }
                          />
                          <YAxis 
                            yAxisId="orders"
                            orientation="right"
                            tickFormatter={(value) => 
                              showSensitiveData 
                                ? value.toFixed(0)
                                : '***'
                            }
                          />
                          <Tooltip
                            labelFormatter={(dateStr) => formatDateWithTimezone(dateStr)}
                            formatter={(value: number, name: string) => {
                              if (name === 'revenue') {
                                return [
                                  showSensitiveData 
                                    ? formatCurrency(value * 100, merchant.currency)
                                    : '***',
                                  'Revenue'
                                ];
                              }
                              return [
                                showSensitiveData ? value.toFixed(0) : '***',
                                'Orders'
                              ];
                            }}
                          />
                          <Area
                            yAxisId="revenue"
                            type="monotone"
                            dataKey="revenue"
                            name="Revenue"
                            fill="hsl(var(--chart-1))"
                            fillOpacity={0.4}
                            stroke="hsl(var(--chart-1))"
                          />
                          <Area
                            yAxisId="orders"
                            type="monotone"
                            dataKey="orders"
                            name="Orders"
                            fill="hsl(var(--chart-2))"
                            fillOpacity={0.4}
                            stroke="hsl(var(--chart-2))"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>
              ) : (
                <div className="h-[240px] flex flex-col items-center justify-center text-muted-foreground">
                  <Ban className="w-12 h-12 mb-4 opacity-50" />
                  <p className="text-sm">No payment data found for this period</p>
                  <p className="text-xs mt-2">
                    Total orders: {merchant.totalOrders} | Total amount: {formatCurrency(merchant.totalAmount, merchant.currency)}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
} 