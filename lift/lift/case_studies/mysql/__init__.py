from lift.case_studies.mysql.mysql_demonstration_rules import MySQLPositiveRule, MySQLNegativeRule, MySQLNoop, \
    MySQLSingleColumn, MySQLPrefixRule

mysql_demo_rules = {
    "full_indexing": MySQLPositiveRule,
    "negative": MySQLNegativeRule,
    "noop": MySQLNoop,
    "single_column": MySQLSingleColumn,
    "prefix_heuristic": MySQLPrefixRule

}