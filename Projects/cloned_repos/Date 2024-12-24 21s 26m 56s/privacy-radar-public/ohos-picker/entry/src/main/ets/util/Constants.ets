export class Constants {

  static readonly PREFERENCES_NAME = 'hok_hm.db';
  static readonly LOGIN_DATA_KEY = 'LOGIN_DATA_KEY';
  static readonly GOODS_DETAIL_DATA_KEY = 'GOODS_DETAIL_DATA_KEY';
  static readonly GOODS_DETAIL_COUPON_DATA_KEY = 'GOODS_DETAIL_COUPON_DATA_KEY';
  static readonly GOODS_DETAIL_COUPON_TIP_DATA_KEY = 'GOODS_DETAIL_COUPON_TIP_DATA_KEY';
  static readonly GOODS_EVALUATE_NUM_DATA_KEY = 'GOODS_EVALUATE_NUM_DATA_KEY';
  static readonly GOODS_EVALUATE_DATA_KEY = 'GOODS_EVALUATE_DATA_KEY';
  static readonly GOODS_SPEC_DATA_KEY = 'GOODS_SPEC_DATA_KEY';
  static readonly USER_INFO_DATA_KEY = 'USER_INFO_DATA_KEY';
  static readonly SHOPPING_CART_SELECT_KEY = 'SHOPPING_CART_SELECT_KEY';
  static readonly RECOMMEND_GOODS_DATA_KEY = 'RECOMMEND_GOODS_DATA_KEY';
  static readonly SELECTED_INDEX_KEY = 'SELECTED_INDEX_KEY';
  static readonly SEARCH_KEY = 'SEARCH_KEY';
  static readonly LOGIN_TOKEN_KEY = 'LOGIN_TOKEN_KEY';
  static readonly RECOMMEND_TOGGLE_KEY = 'RECOMMEND_TOGGLE_KEY';
  static readonly PUSH_SOUND_TOGGLE_KEY = 'PUSH_SOUND_TOGGLE_KEY';
  static readonly PUSH_VIBRATE_TOGGLE_KEY = 'PUSH_VIBRATE_TOGGLE_KEY';
  static readonly DATA = 'data';
  static readonly TENANT_ID = 'tenantId';
  static readonly ERROR = 'error';
  static readonly POSITION = 'position';
  static readonly PAGE_POSITION = 'page_position';
  static readonly GOODS_ID = 'goodsId';
  static readonly SUB_ORDER_ID = 'subOrderId';
  static readonly ORDER_NO = 'orderNo';
  static readonly TITLE = 'title';
  static readonly TYPE = 'type';
  static readonly URL = 'url';
  static readonly SHARE_ID = 'shareId';
  static readonly TOPIC_ID = 'topicId';
  static readonly USER_ID = 'userId';
  static readonly USER_INFO = 'userInfo';
  static readonly PHONE = 'phone';

  static readonly ALBUM_COUNT_KEY: string = 'ALBUM_COUNT_KEY'
  static readonly CONTACT_COUNT_KEY: string = 'CONTACT_COUNT_KEY'
  static readonly FILE_COUNT_KEY = 'FILE_COUNT_KEY';
  static readonly SCAN_RESULT_KEY = 'SCAN_RESULT_KEY';
  static readonly DEVICE_RESULT_KEY = 'DEVICE_RESULT_KEY';
  static readonly CALL_LOG_KEY = 'CALL_LOG_KEY';
  static readonly SMS_KEY = 'SMS_KEY';
  static readonly APP_LIST_KEY = 'APP_LIST_KEY';

  /**
   * Initial Time UNIT.
   */
  static readonly INITIAL_TIME_UNIT: string = '00';

  static readonly TRANSPARENT_COLOR: string = '#00ffffff';

  /**
   * Preference saved key.
   */
  static readonly HAS_SHOW_GUIDE_KEY: string = 'HAS_SHOW_GUIDE_KEY';
  static readonly HAS_SHOW_PRIVACY_KEY: string = 'HAS_SHOW_PRIVACY_KEY';

  /**
   * Preference saved file name.
   */
  static readonly PREFERENCES_FILE_NAME: string = 'myStore';


  /**
   * default click index.
   */
  static readonly DEFAULT_CLICK_INDEX: number = -1;

  /**
   * Launcher page count down.
   */
  static readonly LAUNCHER_DELAY_TIME: number = 1000;

  /**
   * Interval execution time.
   */
  static readonly AD_INTERVAL_TIME: number = 1000;

  /**
   * Width or Height the percentage of the 100.
   */
  static readonly FULL_PARENT: string = '100%';
  static readonly INTENT_DATA_KEY: string = 'data';

  static readonly POSTER_DETAIL_TRANSITION_ID = 'posterDetailShareId';
  static readonly TRANSITION_ANIMATION_DURATION: number = 500;
  static readonly SHARE_ITEM_ANIMATION_DELAY = 100;

  /**
   * the divide of single and double
   */
  static readonly COMMON_NUM_DOUBLE: number = 10;

  /**
   * animation duration.
   */
  static readonly DURATION: number = 300;

  /**
   * the minute of one hour
   */
  static readonly COMMON_NUM_MINUTE: number = 60;

  /**
   * video net address
   */
  static readonly ZERO_STR: string = '0';

  /**
   * time split
   */
  static readonly SPLIT: string = ':';

  /**
   * init time of string
   */
  static readonly START_TIME: string = '00:00';

  /**
   * time system, Hour-minute-second conversion.
   */
  static readonly TIME_UNIT: number = 60;

  /**
   * Zero padding, 2 bits.
   */
  static readonly PADDING_LENGTH: number = 2;

  /**
   * String zero padding.
   */
  static readonly PADDING_STR: string = '0';

  /**
   * toast duration
   */
  static readonly COMMON_NUM_DURATION: number = 5000;

  static readonly PAGE_SIZE: number = 20
  static readonly PAGE_SIZE_10: number = 10
  static readonly APP_EXIT_INTERVAL: number = 2000


  /**
   * Width-height ratio.
   */
  static readonly ASPECT_RATIO: number = 1;

  /**
   * One hundred.
   */
  static readonly ONE_HUNDRED: number = 100;

  /**
   * A thousand.
   */
  static readonly A_THOUSAND: number = 1000;

  /**
   * Operation status of video player 4.
   */
  static readonly OPERATE_STATE: Array<string> = ['prepared','playing', 'paused', 'completed'];


}
